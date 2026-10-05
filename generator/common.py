"""Общие утилиты пайплайна pFAI: чтение данных ImageCAS/ImageCAS-X, граф
центрлинии, корневание дерева, атрибуция вокселей, радиус сосуда.

Все скрипты 01_cycles / 02_overlay / 03_qc / 04_generate / 05_qc_generate
импортируют этот модуль.
Комментарии по-русски, функции — по одной задаче.
"""

from __future__ import annotations

import argparse
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
#  Пути, seed, параметры                                                      #
# --------------------------------------------------------------------------- #

SEED = 42  # базовый seed; от него детерминированно выводятся локальные rng

IMAGECAS_CT = Path("/srv/fast1/y.pchelitsev/datasets/ImageCAS/data")
IMAGECASX = Path("/srv/fast1/y.pchelitsev/datasets/ImageCAS-X")
OUT = Path(__file__).resolve().parent / "out"

# Официальный справочник сегментов ImageCAS-X (совпадает с code/evaluate.py).
SEGMENT_NAMES = {
    1: "LM", 2: "LAD", 3: "LCx", 4: "D1", 5: "D2", 6: "OM1", 7: "OM2",
    8: "IM", 9: "RCA", 10: "R-PDA", 11: "R-PLA", 12: "L-PDA", 13: "L-PLA",
    14: "Other",
}
NAME_TO_ID = {name.upper(): idx for idx, name in SEGMENT_NAMES.items()}
TARGET_NAMES = ("LAD", "LCx", "RCA")          # цели FAI, LM исключён
TARGET_IDS = tuple(NAME_TO_ID[n.upper()] for n in TARGET_NAMES)   # (2, 3, 9)

# (a, b, c, d) в мм: точка начала в [a, b], точка конца в [c, d]. Знак: < 0 —
# расстояние от конца сосуда, >= 0 — от начала (устья; для LAD/LCx — от
# бифуркации LM). Ноль — от начала; для отсчёта ровно от конца берите малое
# отрицательное (напр. -1e-6), т.к. -0.0 намеренно НЕ различается.
WINDOWS = {
    "LAD": (0.0, 5.0, 35.0, 40.0),
    "LCX": (0.0, 5.0, 35.0, 40.0),
    "RCA": (10.0, 15.0, 45.0, 50.0),
}
MIN_FRAGMENT_MM = 20.0     # минимальная геодезическая длина фрагмента
SECTION_HALF_MM = 6.0      # полуразмер сетки поперечного сечения
RADIUS_SANITY = (0.2, 3.0) # допустимый радиус сосуда, мм
MAX_SNAP_MM = 3.0          # порог ухода точки центрлинии от маски (QC)
HU_WINDOW = (-150.0, 600.0)    # окно для серой основы overlay (контраст ~белый)
GREY_GAMMA = 1.0               # гамма серой основы (1.0 = без гамма-коррекции)
# MIP-приоритет по трубке (общий для 02_overlay и 05_qc_generate): на пиксель,
# накрытый проекцией трубки, берётся MIP только внутри неё.
TUBE_K = 5.0                   # радиус трубки = K * локальный радиус сосуда
TUBE_SOFT_MM = 2.0             # ширина мягкого края трубки, мм
# ImageCAS-X отдаёт центрлинии в VTK, где координаты в LPS (как в ITK/DICOM),
# а NIfTI хранит мир в RAS. Разница — знак x и y; без этой поправки центрлиния
# окажется зеркальной относительно маски (проверено: без неё точки вне маски).
VTK_LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])

# Фиксированная палитра по меткам 1..14 (RGB). Три магистрали — интуитивные
# красный/синий/зелёный; соседние по смыслу метки (D1/R-PLA, RCA/R-PDA) разведены
# по тону И яркости, чтобы не путались на одном кадре.
PALETTE = {
    1: (255, 215, 0),     # LM — золотой
    2: (230, 30, 30),     # LAD — красный
    3: (30, 90, 240),     # LCx — ярко-синий
    4: (255, 140, 0),     # D1 — ярко-оранжевый
    5: (150, 40, 200),    # D2 — фиолетовый
    6: (0, 200, 200),     # OM1 — бирюзовый
    7: (255, 0, 200),     # OM2 — маджента
    8: (160, 255, 0),     # IM — лаймовый
    9: (0, 170, 0),       # RCA — зелёный
    10: (0, 0, 140),      # R-PDA — тёмно-синий
    11: (200, 200, 200),  # R-PLA — светло-серый
    12: (120, 0, 0),      # L-PDA — тёмно-красный
    13: (255, 150, 200),  # L-PLA — розовый
    14: (90, 90, 90),     # Other — тёмно-серый
}


def make_rng(seed: int = SEED, *parts) -> np.random.Generator:
    """Детерминированный rng: одинаковые seed и parts дают одинаковую выборку.

    Используем crc32, а не hash(): hash строк рандомизируется между запусками.
    """
    key = zlib.crc32("/".join(str(p) for p in parts).encode("utf-8"))
    return np.random.default_rng([int(seed), int(key)])


# --------------------------------------------------------------------------- #
#  Чтение данных                                                              #
# --------------------------------------------------------------------------- #

def ct_path(scan_id: str, root: Path = IMAGECAS_CT) -> Path:
    return Path(root) / f"{scan_id}.img.nii.gz"


def mask_path(scan_id: str, root: Path = IMAGECASX) -> Path:
    return Path(root) / "segmentations" / f"{scan_id}.coronary.nii.gz"


def centerline_path(scan_id: str, tree: str, root: Path = IMAGECASX) -> Path:
    return Path(root) / "centerlines" / f"{scan_id}.coronary_{tree}_centerline.vtk"


def filelist(split: str, root: Path = IMAGECASX) -> list[str]:
    """ID сканов сплита из filelist/<split>.txt (train/val/test/exclude)."""
    text = (Path(root) / "filelist" / f"{split}.txt").read_text().split()
    return [s for s in text if s]


def read_ct(scan_id: str, root: Path = IMAGECAS_CT) -> tuple[np.ndarray, np.ndarray]:
    """КТ-объём в HU (float32) и его affine."""
    img = nib.load(str(ct_path(scan_id, root)))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine


def ct_affine(scan_id: str, root: Path = IMAGECAS_CT) -> np.ndarray:
    """Affine КТ без чтения данных: nibabel грузит только заголовок.

    Нужен там, где сверяется геометрия КТ и маски (`check_geometry`), но сам
    объём КТ не используется — экономит полный I/O объёма на каждый скан.
    """
    return np.asarray(nib.load(str(ct_path(scan_id, root))).affine, dtype=float)


@lru_cache(maxsize=4)
def read_mask(scan_id: str, root: Path = IMAGECASX) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Маска сегментов (uint8), affine и спейсинг (мм)."""
    img = nib.load(str(mask_path(scan_id, root)))
    data = np.asanyarray(img.dataobj).astype(np.uint8)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, img.affine, zooms


@dataclass
class Centerline:
    """Центрлиния одного дерева в мировых мм (RAS) с поточечными метками."""

    scan_id: str
    tree: str                       # "left" или "right"
    points: np.ndarray              # (N, 3) мм
    labels: np.ndarray              # (N,) int
    names: np.ndarray               # (N,) str
    is_start: np.ndarray            # (N,) bool
    is_end: np.ndarray              # (N,) bool
    is_branch: np.ndarray           # (N,) bool
    cells: list[np.ndarray]         # ячейки-полилинии (индексы точек)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def roots(self) -> list[int]:
        idx = np.flatnonzero(self.is_start)
        return idx.tolist() if len(idx) else [0]


@lru_cache(maxsize=8)
def read_centerline(scan_id: str, tree: str, root: Path = IMAGECASX) -> "Centerline":
    """Читает legacy VTK центрлинии, переводит координаты LPS -> RAS (мм)."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(centerline_path(scan_id, tree, root)))
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    poly = reader.GetOutput()

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float) * VTK_LPS_TO_RAS

    pdata = poly.GetPointData()

    def flag(name: str) -> np.ndarray:
        arr = pdata.GetAbstractArray(name)
        return vtk_to_numpy(arr).astype(bool) if arr is not None else np.zeros(len(points), bool)

    labels_arr = pdata.GetAbstractArray("segment_label")
    labels = vtk_to_numpy(labels_arr).astype(int)

    sarr = vtk.vtkStringArray.SafeDownCast(pdata.GetAbstractArray("segment_name"))
    if sarr is not None:
        names = np.array([sarr.GetValue(i) for i in range(sarr.GetNumberOfTuples())], dtype=object)
    else:
        names = np.array([SEGMENT_NAMES.get(int(v), str(v)) for v in labels], dtype=object)

    cells: list[np.ndarray] = []
    lines = poly.GetLines()
    if lines is not None and lines.GetNumberOfCells():
        lines.InitTraversal()
        ids = vtk.vtkIdList()
        while lines.GetNextCell(ids):
            cell = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
            if len(cell) > 1:
                cells.append(np.array(cell, dtype=int))

    return Centerline(scan_id, tree, points, labels, names,
                      flag("start_points"), flag("end_points"), flag("branch_points"), cells)


def check_geometry(affine_ct: np.ndarray, affine_mask: np.ndarray) -> None:
    """Падает, если геометрия КТ и маски не совпадает (ресемплинг не делаем)."""
    if not np.allclose(affine_ct, affine_mask, atol=1e-4):
        raise SystemExit("геометрия КТ и маски не совпадает — это ошибка данных")


# --------------------------------------------------------------------------- #
#  Переводы координат                                                         #
# --------------------------------------------------------------------------- #

def to_world(affine: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """Индексы вокселей -> мировые мм по affine (сетка нативная, без ресемплинга)."""
    vox = np.asarray(vox, dtype=float)
    return vox @ affine[:3, :3].T + affine[:3, 3]


def to_voxel(affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Мировые мм -> непрерывные индексы вокселей (обратный affine)."""
    inv = np.linalg.inv(affine)
    return np.asarray(world, dtype=float) @ inv[:3, :3].T + inv[:3, 3]


def world_to_voxel_index(affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Мировые мм -> ближайший целый индекс вокселя (для выборки значений)."""
    return np.rint(to_voxel(affine, world)).astype(int)


# --------------------------------------------------------------------------- #
#  Граф центрлинии                                                            #
# --------------------------------------------------------------------------- #

def _cell_edges(cl: Centerline):
    """Все рёбра ячеек: (a, b, длина_мм)."""
    for cell in cl.cells:
        for a, b in zip(cell[:-1], cell[1:]):
            yield int(a), int(b), float(np.linalg.norm(cl.points[a] - cl.points[b]))


def build_graph(cl: Centerline) -> nx.Graph:
    """Простой неориентированный граф (для корневания, путей, потомков).

    Простой Graph схлопывает параллельные рёбра (одна пара узлов из разных ячеек);
    это допустимо для обхода дерева, но для анализа циклов нужен build_multigraph.
    """
    g = nx.Graph()
    g.add_nodes_from(range(cl.n_points))
    for a, b, w in _cell_edges(cl):
        g.add_edge(a, b, weight=w)
    return g


def build_multigraph(cl: Centerline) -> tuple[nx.MultiGraph, nx.Graph]:
    """(мультиграф, простой граф) для анализа циклов.

    Мультиграф сохраняет параллельные рёбра (одна пара узлов из разных ячеек),
    простой граф их схлопывает. Оба строятся за один проход по ячейкам.
    """
    multi = nx.MultiGraph()
    simple = nx.Graph()
    multi.add_nodes_from(range(cl.n_points))
    simple.add_nodes_from(range(cl.n_points))
    for a, b, w in _cell_edges(cl):
        multi.add_edge(a, b, weight=w)
        if simple.has_edge(a, b):
            simple[a][b]["weight"] = min(simple[a][b]["weight"], w)
        else:
            simple.add_edge(a, b, weight=w)
    return multi, simple


@dataclass
class Rooted:
    """Дерево, корневанное в устье: родители, дети, порядок обхода и arc."""

    cl: Centerline
    graph: nx.Graph
    root: int
    roots: list[int]
    parent: dict
    children: dict
    order: list
    arc: dict                 # узел -> расстояние от устья по графу, мм

    def descendants(self, node: int) -> set[int]:
        """Все потомки узла (без него самого)."""
        out: set[int] = set()
        stack = list(self.children.get(node, []))
        while stack:
            v = stack.pop()
            if v in out:
                continue
            out.add(v)
            stack.extend(self.children.get(v, []))
        return out


def root_tree(cl: Centerline, graph: nx.Graph | None = None) -> Rooted:
    """Корневание: BFS от всех устьев (start_points), arc — кратчайший путь в мм.

    Устьев может быть несколько: в левой системе без общего ствола LAD и LCx
    имеют отдельные ostium. Поэтому стартуем BFS сразу из всех помеченных точек
    (как делает и ImageCAS-X), а arc считаем multi-source Дейкстрой.
    """
    g = graph if graph is not None else build_graph(cl)
    roots = cl.roots

    # Обход в ширину: parent/children задают «вниз по дереву» (к дистальным концам).
    parent: dict[int, int | None] = {r: None for r in roots}
    children: dict[int, list[int]] = defaultdict(list)
    order: list[int] = list(roots)
    head = 0
    while head < len(order):
        u = order[head]
        head += 1
        for v in g.neighbors(u):
            if v not in parent:
                parent[v] = u
                children[u].append(v)
                order.append(v)

    # arc — длина дуги от ближайшего устья, мм. Недостижимые точки в dict не попадут.
    arc = nx.multi_source_dijkstra_path_length(g, roots, weight="weight")
    root = roots[0]
    return Rooted(cl, g, root, roots, parent, dict(children), order, arc)


def vessel_nodes(rooted: Rooted, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Узлы метки, отсортированные по arc от устья, и их arc.

    Отбрасываем точки, недостижимые от устья (arc = inf): такие бывают в
    оторванных компонентах без start_points и к сосуду не относятся.
    """
    idx = np.flatnonzero(rooted.cl.labels == label)
    if len(idx) == 0:
        return idx, np.zeros(0)
    arc = np.array([rooted.arc.get(int(i), np.inf) for i in idx])
    order = np.argsort(arc, kind="stable")
    idx = idx[order]
    arc = arc[order]
    return idx[np.isfinite(arc)], arc[np.isfinite(arc)]


def path_between(rooted: Rooted, start: int, end: int) -> list[int]:
    """Путь по дереву между узлами (единственный при отсутствии циклов)."""
    return nx.shortest_path(rooted.graph, start, end, weight="weight")


# --------------------------------------------------------------------------- #
#  Атрибуция воксель маски -> точка центрлинии                                #
# --------------------------------------------------------------------------- #

@dataclass
class Attribution:
    """Соответствие каждого меченого вокселя ближайшей точке центрлинии."""

    voxels: np.ndarray          # (M, 3) индексы вокселей
    labels: np.ndarray          # (M,) метка вокселя
    node: np.ndarray            # (M,) глобальный индекс точки центрлинии
    dist_mm: np.ndarray         # (M,) расстояние до неё
    node_points: np.ndarray     # (N, 3) все точки центрлинии (left + right)
    node_labels: np.ndarray     # (N,) метки точек
    offsets: dict[str, int]     # дерево -> сдвиг глобальной нумерации
    centers: dict[str, Centerline] = field(default_factory=dict)

    def global_nodes(self, tree: str, local: np.ndarray) -> np.ndarray:
        return local + self.offsets[tree]


def attribute_voxels(mask: np.ndarray, affine: np.ndarray,
                     centerlines: dict[str, Centerline]) -> Attribution:
    """Для каждого вокселя маски (метки 1..14) — ближайшая точка центрлинии.

    KD-tree строится по точкам ОБОИХ деревьев (left + right) в мировых мм, поэтому
    воксель всегда находит ближайшую точку своего сосуда. Это ключ к выделению
    фрагмента: воксели боковых ветвей ближе к своим точкам и в целевой путь не
    попадут. Согласие метки вокселя и метки ближайшей точки — 100% на проверках.
    """
    voxels = np.argwhere(mask > 0)
    world = to_world(affine, voxels)

    # Глобальная нумерация узлов: сначала left, затем right (offsets — для отката).
    parts, labels_parts, offsets = [], [], {}
    shift = 0
    for tree, cl in centerlines.items():
        offsets[tree] = shift
        parts.append(cl.points)
        labels_parts.append(cl.labels)
        shift += cl.n_points
    node_points = np.vstack(parts)
    node_labels = np.concatenate(labels_parts)

    dist, node = cKDTree(node_points).query(world)
    return Attribution(voxels, mask[voxels[:, 0], voxels[:, 1], voxels[:, 2]],
                       node, dist, node_points, node_labels, offsets, dict(centerlines))


# --------------------------------------------------------------------------- #
#  Радиус сосуда                                                              #
# --------------------------------------------------------------------------- #

def node_tangents(rooted: "Rooted") -> np.ndarray:
    """Касательные во всех точках дерева по его рёбрам.

    Точки в VTK не упорядочены вдоль сосуда (порядок задают ячейки), поэтому
    брать соседей по индексу массива нельзя. Идём по родителю и ребёнку; на
    бифуркации выбираем ребёнка, наиболее коллинеарного входу (продолжение).
    Детей сортируем — иначе порядок из кэша и из свежего чтения мог бы дать
    разный выбор при почти равных углах.
    """
    cl = rooted.cl
    out = np.zeros((cl.n_points, 3))
    for i in range(cl.n_points):
        p = cl.points[i]
        par = rooted.parent.get(i)
        ch = sorted(rooted.children.get(i, []))
        if par is not None and ch:
            base = p - cl.points[par]
            nb = base / (np.linalg.norm(base) or 1.0)
            c = ch[int(np.argmax((cl.points[ch] - p) @ nb))]
            t = cl.points[c] - cl.points[par]
        elif par is not None:
            t = p - cl.points[par]
        elif ch:
            t = cl.points[ch[0]] - p
        else:
            t = np.array([1.0, 0.0, 0.0])
        out[i] = t / (np.linalg.norm(t) or 1.0)
    return out


def _section_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(tangent, float)
    t = t / (np.linalg.norm(t) or 1.0)
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(t, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    return t, e1, e2


@dataclass
class VesselMask:
    """Воксели одного сосуда (метки) как KD-tree, для быстрого радиуса по сечению."""

    points: np.ndarray          # (M, 3) мировые мм
    tree: cKDTree
    voxel_volume: float
    zooms: tuple

    def section_radius(self, point: np.ndarray, tangent: np.ndarray,
                       half_mm: float = SECTION_HALF_MM) -> tuple[float, float]:
        """Радиус по площади поперечного сечения (оценка Кавалери).

        Считаем объём вокселей в тонком цилиндре радиуса half_mm и толщины w
        вокруг точки; A = N * voxel_volume / w, r = sqrt(A / pi). Медиана по
        нескольким w гасит шум дискретизации. В отличие от объединения
        подслоёв, не раздувает площадь косого сосуда.
        """
        t, _, _ = _section_basis(tangent)
        w0 = max(float(min(self.zooms)), 0.25)
        reach = float(np.hypot(half_mm, 2.0 * w0))
        idx = self.tree.query_ball_point(point, reach)
        if not idx:
            return 0.0, 0.0
        rel = self.points[idx] - point
        axial = rel @ t
        areas = []
        for w in (w0, 1.5 * w0, 2.0 * w0):
            n = int(np.count_nonzero(np.abs(axial) <= w / 2))
            if n:
                areas.append(n * self.voxel_volume / w)
        if not areas:
            return 0.0, 0.0
        area = float(np.median(areas))
        return float(np.sqrt(area / np.pi)), area


def vessel_mask(mask: np.ndarray, affine: np.ndarray, zooms: tuple, labels) -> VesselMask:
    """Строит VesselMask по вокселям маски с указанными метками.

    np.isin по всему объёму — относительно дорого, поэтому маски по меткам
    строятся один раз на скан (в prepare_scan) и кладутся в кэш.
    """
    sel = np.isin(mask, list(labels))
    points = to_world(affine, np.argwhere(sel))
    return VesselMask(points, cKDTree(points), float(np.prod(zooms)), tuple(zooms))


def edt_radius_all(mask: np.ndarray, affine: np.ndarray, zooms: tuple,
                   points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EDT-радиус в ближайшем вокселе просвета (объединение всех меток > 0).

    Один расчёт EDT на скан: на стыках меток (LM-LAD) просвет непрерывен, поэтому
    union не даёт ложного обрыва, как EDT по отдельной метке.
    """
    binary = mask > 0
    if not binary.any():
        return np.zeros(len(points)), np.full(len(points), np.inf)
    coords = np.argwhere(binary)
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, binary.shape)
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    edt = distance_transform_edt(binary[sl], sampling=zooms)

    dist, j = cKDTree(to_world(affine, coords)).query(points)
    near = coords[j] - lo
    return edt[near[:, 0], near[:, 1], near[:, 2]], dist


def tree_radius(rooted: "Rooted", vms: dict, r_edt: np.ndarray, snap: np.ndarray) -> dict:
    """Радиус по сечению и EDT во всех точках дерева (по метке каждой точки).

    vms — заранее собранные маски сосудов по меткам; r_edt/snap — заранее
    посчитанный EDT на все точки скана (чтобы не пересчитывать на каждое дерево).
    """
    cl = rooted.cl
    n = cl.n_points
    tang = node_tangents(rooted)
    r_sec = np.zeros(n)
    for label in sorted(set(cl.labels.tolist())):
        if label == 0 or label not in vms:
            continue
        vm = vms[label]
        for i in np.flatnonzero(cl.labels == label):
            r_sec[i] = vm.section_radius(cl.points[i], tang[i])[0]
    arc = np.array([rooted.arc.get(i, np.nan) for i in range(n)])
    return {"points": cl.points, "labels": cl.labels, "arc": arc,
            "radius_section": r_sec, "radius_edt": r_edt, "snap_mm": snap, "tangent": tang}


# --------------------------------------------------------------------------- #
#  Дисковый кэш предобработки скана                                           #
# --------------------------------------------------------------------------- #

CACHE_DIR = OUT / "cache"
TREES = ("left", "right")


@dataclass
class PreparedScan:
    """Всё тяжёлое для скана: маска, центрлинии, деревья, радиусы, атрибуция."""

    scan_id: str
    mask: np.ndarray
    affine: np.ndarray
    zooms: tuple
    lines: dict
    centered: dict
    radii: dict
    attr: "Attribution"
    vms: dict


def _pack_cells(cells: list) -> tuple[np.ndarray, np.ndarray]:
    """Ячейки переменной длины -> плоский массив + смещения (для npz)."""
    flat = np.concatenate(cells) if cells else np.zeros(0, int)
    off = np.array([0] + [len(c) for c in cells]).cumsum()
    return flat, off


def _unpack_cells(flat: np.ndarray, off: np.ndarray) -> list:
    """Обратно из плоского массива и смещений в список ячеек."""
    return [flat[off[i]:off[i + 1]].copy() for i in range(len(off) - 1)]


def _centerline_from_arrays(scan_id, tree, z) -> Centerline:
    """Восстанавливает Centerline из кэша.

    is_end/is_branch и names не сохраняем: они нужны только при чтении/QC, а не
    для генерации; names синтезируем из labels.
    """
    labels = z[f"{tree}_labels"].astype(int)
    return Centerline(
        scan_id, tree, z[f"{tree}_points"], labels,
        np.array([SEGMENT_NAMES.get(int(v), str(v)) for v in labels], dtype=object),
        z[f"{tree}_start"].astype(bool), np.zeros(len(labels), bool),
        np.zeros(len(labels), bool), _unpack_cells(z[f"{tree}_cells"], z[f"{tree}_cell_off"]))


def _rooted_from_arrays(cl: Centerline, z, tree: str) -> Rooted:
    """Восстанавливает Rooted из кэша: граф из рёбер, дети из parent, arc из файла.

    order здесь — просто 0..N-1, а не BFS-порядок: он нигде не используется для
    генерации. Если понадобится (например, для QC недостижимых), сохраним реальный.
    """
    g = nx.Graph()
    g.add_nodes_from(range(cl.n_points))
    for u, v, w in zip(z[f"{tree}_eu"], z[f"{tree}_ev"], z[f"{tree}_ew"]):
        g.add_edge(int(u), int(v), weight=float(w))
    roots = cl.roots
    parent = {i: (None if p < 0 else int(p)) for i, p in enumerate(z[f"{tree}_parent"])}
    children: dict[int, list[int]] = defaultdict(list)
    for i, p in parent.items():
        if p is not None:
            children[p].append(i)
    arc = {i: float(a) for i, a in enumerate(z[f"{tree}_arc"])}
    return Rooted(cl, g, roots[0], roots, parent, dict(children), list(range(cl.n_points)), arc)


def cache_path(scan_id: str) -> Path:
    return CACHE_DIR / f"{scan_id}.npz"


def _save_cache(ps: PreparedScan) -> None:
    """Пишет всё тяжёлое в один сжатый npz (маска, деревья, радиусы, атрибуция)."""
    arrays = {"mask": ps.mask, "affine": ps.affine, "zooms": np.array(ps.zooms),
              "labels_present": np.array(sorted(ps.vms), dtype=int),
              "attr_vox": ps.attr.voxels, "attr_node": ps.attr.node,
              "attr_dist": ps.attr.dist_mm, "attr_labels": ps.attr.labels}
    # Мировые точки масок по меткам: на загрузке не пересобираем np.isin по объёму.
    for lab, vm in ps.vms.items():
        arrays[f"vm_{lab}"] = vm.points
    for tree, rooted in ps.centered.items():
        cl = rooted.cl
        # Рёбра и плоские ячейки — чтобы восстановить граф и топологию без VTK.
        eu, ev, ew = [], [], []
        for u, v, w in _cell_edges(cl):
            eu.append(u); ev.append(v); ew.append(w)
        flat, off = _pack_cells(cl.cells)
        parent = np.array([-1 if rooted.parent.get(i) is None else rooted.parent[i]
                           for i in range(cl.n_points)])
        arrays.update({
            f"{tree}_points": cl.points, f"{tree}_labels": cl.labels,
            f"{tree}_start": cl.is_start, f"{tree}_cells": flat, f"{tree}_cell_off": off,
            f"{tree}_eu": np.array(eu, int), f"{tree}_ev": np.array(ev, int),
            f"{tree}_ew": np.array(ew, float), f"{tree}_parent": parent,
            # arc у недостижимых от устья точек отсутствует -> NaN, а не KeyError.
            f"{tree}_arc": np.array([rooted.arc.get(i, np.nan) for i in range(cl.n_points)]),
            f"{tree}_tangent": ps.radii[tree]["tangent"],
            f"{tree}_rsec": ps.radii[tree]["radius_section"],
            f"{tree}_redt": ps.radii[tree]["radius_edt"],
            f"{tree}_snap": ps.radii[tree]["snap_mm"]})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(ps.scan_id), **arrays)


def _load_cache(scan_id: str) -> PreparedScan:
    """Читает npz и собирает PreparedScan (маски — из сохранённых точек)."""
    z = np.load(cache_path(scan_id))
    mask = z["mask"]; affine = z["affine"]; zooms = tuple(float(v) for v in z["zooms"])
    lines, centered, radii = {}, {}, {}
    for tree in TREES:
        cl = _centerline_from_arrays(scan_id, tree, z)
        lines[tree] = cl
        centered[tree] = _rooted_from_arrays(cl, z, tree)
        radii[tree] = {"tangent": z[f"{tree}_tangent"], "radius_section": z[f"{tree}_rsec"],
                       "radius_edt": z[f"{tree}_redt"], "snap_mm": z[f"{tree}_snap"],
                       "points": cl.points, "labels": cl.labels, "arc": z[f"{tree}_arc"]}
    present = [int(v) for v in z["labels_present"]]
    vv = float(np.prod(zooms))
    vms = {lab: VesselMask(z[f"vm_{lab}"], cKDTree(z[f"vm_{lab}"]), vv, zooms)
           for lab in present}
    # Порядок точек/меток и сдвиги должны совпадать с attribute_voxels: left, затем right.
    n_left = lines["left"].n_points
    pts = np.vstack([lines[t].points for t in TREES])
    labs = np.concatenate([lines[t].labels for t in TREES])
    attr = Attribution(z["attr_vox"], z["attr_labels"], z["attr_node"], z["attr_dist"],
                       pts, labs, {"left": 0, "right": n_left}, lines)
    return PreparedScan(scan_id, mask, affine, zooms, lines, centered, radii, attr, vms)


def prepare_scan(scan_id: str, use_cache: bool = True) -> PreparedScan:
    """Читает/считает предобработку скана, кэшируя результат на диск.

    Дорогое — радиусы по сечению (~10 с) и EDT. При use_cache=True и наличии
    out/cache/<scan>.npz тяжёлое не пересчитывается. Кэш обязан быть
    детерминированным: node_tangents сортирует детей, иначе результат из кэша
    мог бы отличаться от свежего.
    """
    if use_cache and cache_path(scan_id).exists():
        return _load_cache(scan_id)

    mask, affine, zooms = read_mask(scan_id)
    lines = {t: read_centerline(scan_id, t) for t in TREES}
    attr = attribute_voxels(mask, affine, lines)
    centered = {t: root_tree(cl, build_graph(cl)) for t, cl in lines.items()}

    # Маски по меткам и EDT по всему просвету — один раз на скан, а не на дерево.
    labels_present = sorted(int(v) for v in np.unique(mask) if v)
    vms = {lab: vessel_mask(mask, affine, zooms, [lab]) for lab in labels_present}
    pts = np.vstack([lines[t].points for t in TREES])
    r_edt_all, snap_all = edt_radius_all(mask, affine, zooms, pts)
    radii = {}
    for t in TREES:
        n = lines[t].n_points
        start = 0 if t == "left" else lines["left"].n_points
        radii[t] = tree_radius(centered[t], vms,
                               r_edt_all[start:start + n], snap_all[start:start + n])

    ps = PreparedScan(scan_id, mask, affine, zooms, lines, centered, radii, attr, vms)
    if use_cache:
        _save_cache(ps)
    return ps


# --------------------------------------------------------------------------- #
#  Запись NIfTI                                                               #
# --------------------------------------------------------------------------- #

def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_mask_nifti(array: np.ndarray, affine: np.ndarray, zooms: tuple,
                    path: Path, dtype=np.uint8) -> Path:
    """Сохраняет скалярный объём с геометрией исходного скана."""
    path = _ensure_parent(path)
    img = nib.Nifti1Image(np.ascontiguousarray(array.astype(dtype)), affine)
    img.header.set_zooms(tuple(float(z) for z in zooms))
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return path


def save_rgb_nifti(rgb: np.ndarray, affine: np.ndarray, zooms: tuple, path: Path) -> Path:
    """Сохраняет цветной объём (X, Y, Z, 3) uint8 как RGB24 NIfTI.

    nibabel пишет RGB только структурным dtype [('R',u1),('G',u1),('B',u1)]
    и shape (X, Y, Z, 1); иначе WriterError про non-numeric types.
    """
    path = _ensure_parent(path)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    struct = rgb.view(np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")]))
    struct = struct.reshape(*rgb.shape[:3], 1)

    header = nib.Nifti1Header()
    header.set_data_dtype("RGB")
    header.set_data_shape(struct.shape)
    header.set_zooms(tuple(float(z) for z in zooms) + (1.0,))
    header.set_xyzt_units("mm")
    nib.save(nib.Nifti1Image(struct, affine, header=header), str(path))
    return path


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Цветной объём по меткам 1..14 (фон чёрный)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in PALETTE.items():
        rgb[mask == label] = color
    return rgb


def save_rgb_tiff(rgb: np.ndarray, zooms: tuple, path: Path) -> Path:
    """Сохраняет RGB-объём (X,Y,Z,3) как multi-page TIFF для ImageJ/Fiji.

    NIfTI RGB24 читается ImageJ/Fiji плохо (соляризация/битые каналы). TIFF с
    флагом imagej=True и прописанным разрешением открывается корректно: ImageJ
    сам собирает RGB-гиперстек. Переставляем оси под ImageJ: (Z, Y, X, 3).
    """
    import tifffile

    path = _ensure_parent(path)
    arr = np.transpose(np.asarray(rgb, dtype=np.uint8), (2, 1, 0, 3))
    res = (1.0 / float(zooms[0]), 1.0 / float(zooms[1]))   # пикселей на мм по x,y
    # axes='ZYXS' (порядок ImageJ TZCYXS): иначе tifffile примет Z за каналы.
    tifffile.imwrite(str(path), arr, imagej=True, resolution=res,
                     metadata={"axes": "ZYXS", "spacing": float(zooms[2]), "unit": "mm"},
                     compression="zlib")
    return path


def save_labels_tiff(mask: np.ndarray, zooms: tuple, path: Path) -> Path:
    """Сохраняет скалярные метки как multi-page TIFF для ImageJ/Fiji."""
    import tifffile

    path = _ensure_parent(path)
    arr = np.transpose(np.asarray(mask, dtype=np.uint8), (2, 1, 0))
    res = (1.0 / float(zooms[0]), 1.0 / float(zooms[1]))
    tifffile.imwrite(str(path), arr, imagej=True, resolution=res,
                     metadata={"axes": "ZYX", "spacing": float(zooms[2]), "unit": "mm"},
                     compression="zlib")
    return path


def paint_points(shape: tuple, affine: np.ndarray, points: np.ndarray,
                 radius_vox: int = 0) -> np.ndarray:
    """Булев объём с отмеченными точками (с опциональной дилатацией).

    Дилатация нужна, чтобы линия толщиной в один воксель была видна при просмотре
    (иначе на проекциях/в ImageJ она почти исчезает).
    """
    out = np.zeros(shape, dtype=bool)
    vox = world_to_voxel_index(affine, points)
    inside = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
    vc = vox[inside]
    out[vc[:, 0], vc[:, 1], vc[:, 2]] = True
    if radius_vox > 0:
        from scipy.ndimage import binary_dilation
        out = binary_dilation(out, np.ones((3, 3, 3), bool), iterations=radius_vox)
    return out


def grey_from_hu(ct: np.ndarray, window: tuple = HU_WINDOW,
                 gamma: float = GREY_GAMMA) -> np.ndarray:
    """КТ -> float32 [0, 1] по окну HU с гамма-коррекцией.

    Гамма < 1 поднимает тусклые средние тона (мягкие ткани), не выбивая яркие
    контрастные сосуды/кость в насыщение — иначе скан в overlay выглядит тёмным.
    """
    lo, hi = window
    x = np.clip((ct - lo) / (hi - lo), 0.0, 1.0)
    return np.power(x, gamma, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Анатомические проекции (общий код 02_overlay и 05_qc_generate)             #
# --------------------------------------------------------------------------- #

# Панели: (заголовок, ось проекции drop, вертикальная ось row, горизонтальная
# ось col, подпись вертикали, подпись горизонтали). Ориентация анатомическая:
# сагиттальная/корональная с вертикалью S/I (z), аксиальная с вертикалью A/P (y).
PROJECTION_PANELS = (
    ("Сагиттальная (вид сбоку, вдоль x)", 0, 2, 1, "S/I (z)", "A/P (y)"),
    ("Корональная (вид спереди, вдоль y)", 1, 2, 0, "S/I (z)", "R/L (x)"),
    ("Аксиальная (вид сверху, вдоль z)", 2, 1, 0, "A/P (y)", "R/L (x)"),
)


def orient_mip(volume, drop: int, row: int, col: int, affine) -> tuple:
    """MIP вдоль оси drop, развёрнутый так, чтобы строки = row, столбцы = col.

    Возвращает (изображение, flip_row, flip_col): флаги нужны, чтобы так же
    перевернуть координаты центрлиний. Переворот по оси с отрицательным
    диагональным элементом affine даёт рост мировой координаты вверх/вправо.
    Поддерживает и скалярный (X,Y,Z), и цветной (X,Y,Z,3) объём.
    """
    rem = [i for i in range(3) if i != drop]
    order = (rem.index(row), rem.index(col))
    proj = volume.max(axis=drop)
    if volume.ndim == 4:
        order = order + (2,)          # после MIP цветовой канал стал последним (ось 2)
    proj = np.transpose(proj, order)
    flip_r = bool(affine[row, row] < 0)
    flip_c = bool(affine[col, col] < 0)
    if flip_r:
        proj = proj[::-1]
    if flip_c:
        proj = proj[:, ::-1]
    return proj, flip_r, flip_c


def voxel_in_view(points, affine, row, col, shape, flip_r, flip_c):
    """Мировые точки -> координаты (col, row) в развёрнутом изображении."""
    vox = to_voxel(affine, points)
    c = vox[:, col].copy()
    r = vox[:, row].copy()
    if flip_r:
        r = (shape[0] - 1) - r
    if flip_c:
        c = (shape[1] - 1) - c
    return c, r


def view_limits(mask, affine, margin_mm: float = 25.0) -> dict:
    """Границы просмотра по каждой оси: bbox маски + запас, с учётом флипов.

    Показываем область сердца, а не весь кадр: так и скан «крупнее», и ярче.
    """
    idx = np.argwhere(mask > 0)
    n = np.array(mask.shape)
    lo = idx.min(axis=0) if idx.size else np.zeros(3, int)
    hi = idx.max(axis=0) if idx.size else n - 1
    spacing = np.abs(np.diag(affine))[:3]
    limits = {}
    for a in range(3):
        m = int(np.ceil(margin_mm / spacing[a])) if spacing[a] > 0 else 0
        a0, a1 = max(0, int(lo[a]) - m), min(n[a] - 1, int(hi[a]) + m)
        limits[a] = ((n[a] - 1 - a1, n[a] - 1 - a0) if affine[a, a] < 0 else (a0, a1))
    return limits


def tube_weight(mask, affine, zooms, lines, k: float, soft_mm: float) -> np.ndarray:
    """Мягкий вес [0,1] по вокселям: 1 внутри трубки вокруг центрлиний.

    Для каждой точки центрлинии берём радиус (EDT) и закрашиваем мягкую сферу
    радиусом k*r со спадающим краем, но не шире soft_mm (иначе у тонких сосудов
    вес в центре не дойдёт до 1). Объединяем по всем сосудам через максимум.
    """
    points = np.vstack([cl.points for cl in lines.values()])
    radius, _ = edt_radius_all(mask, affine, zooms, points)
    tube_r = np.clip(k * radius, 0.8, 12.0)

    shape = np.array(mask.shape)
    spacing = np.abs(np.diag(affine))[:3]
    nodes_vox = world_to_voxel_index(affine, points)
    weight = np.zeros(mask.shape, dtype=np.float32)
    for p, r in zip(nodes_vox, tube_r):
        # Мягкость на узел: не шире половины трубки, иначе у тонких сосудов
        # вес в центре просвета не дойдёт до 1.
        soft = min(soft_mm, 0.5 * r)
        rad = np.ceil(r / spacing).astype(int)
        lo = np.maximum(p - rad, 0)
        hi = np.minimum(p + rad + 1, shape)
        if np.any(hi <= lo):
            continue
        g = [(np.arange(lo[a], hi[a]) - p[a]) * spacing[a] for a in range(3)]
        dist = np.sqrt(g[0][:, None, None] ** 2 + g[1][None, :, None] ** 2
                       + g[2][None, None, :] ** 2)
        soft_w = np.clip((r - dist) / soft, 0.0, 1.0).astype(np.float32)
        sub = weight[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.maximum(sub, soft_w, out=sub)
    return weight


def parse_common_args(description: str) -> argparse.Namespace:
    """Общие аргументы командной строки для скриптов пайплайна.

    --strict-windows читает только 04; --no-cache — 04 и 05; в остальных скриптах
    они игнорируются (argparse их принимает, код не читает).
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--split", default="test", help="сплит ImageCAS-X")
    ap.add_argument("--ids", nargs="*", default=None, help="явные ID сканов")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число сканов")
    ap.add_argument("--out", type=Path, default=OUT, help="каталог вывода")
    ap.add_argument("--no-cache", action="store_true", help="не использовать дисковый кэш")
    ap.add_argument("--strict-windows", action="store_true",
                    help="пропускать сосуды, где окна A/B/C/D не влезают без ужима")
    return ap.parse_args()


def resolve_scan_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.ids) if args.ids else filelist(args.split)
    if args.limit:
        ids = ids[: args.limit]
    return ids
