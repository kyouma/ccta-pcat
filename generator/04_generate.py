"""04 — генерация примеров: 3 фрагмента на скан (LAD, LCx, RCA).

Для каждого скана и каждого целевого сосуда:
  1. на ветке выбирается случайная точка старта в окне [A, B] мм;
  2. среди потомков старта — случайная точка конца в окне [C, D] мм;
  3. окна задаются знаковыми смещениями (`WINDOWS`): >= 0 — от начала сосуда
     (устья; для LAD/LCx — от бифуркации LM), < 0 — от его конца;
  4. точки клика смещаются от центрлинии в плоскости сечения (случайный воксель
     маски, через который проходит плоскость);
  5. фрагмент маски — воксели, ближайшая точка центрлинии которых лежит на
     выбранном отрезке пути;
  6. считаются радиусы по сечению (и EDT как cross-check) в точках.

Выход:
  out/generate/fragments/<scan>_<vessel>.nii.gz   маска фрагмента (0/1)
  out/generate/samples/<scan>_<vessel>.json        клики, центрлиния, радиусы
  out/generate/radii/<scan>_<tree>.npz             радиус в каждой точке дерева
  out/generate/points.jsonl                        сводка по примерам (id, длина, воксели)
  out/generate/qc/<scan>_<vessel>.png              QC-картинка с легендой

Запуск:
    python 04_generate.py --ids 961
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label, median_filter
from tqdm import tqdm

import common
from common import (MIN_FRAGMENT_MM, SECTION_HALF_MM, SEGMENT_NAMES, TARGET_IDS, WINDOWS,
                    Attribution, Rooted, path_between, prepare_scan, save_mask_nifti,
                    section_area, section_coverage, vessel_nodes)

MAX_ATTEMPTS = 30          # попыток выбрать допустимый фрагмент на сосуд
CLICK_REGION_COV = 0.5     # порог cov, задающий просвет в плоскости среза
CLICK_DEPTH_FRAC = 0.3     # отступ клика от границы как доля локального радиуса
CLICK_DEPTH_MIN_VOX = 0.5  # ... но не меньше стольких вокселей (min(zooms))
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


# --------------------------------------------------------------------------- #
#  Выбор точек                                                                #
# --------------------------------------------------------------------------- #

def _offset_to_arc(offset: float, arc_root: float, arc_tip: float) -> float:
    """Знаковое смещение вдоль сосуда (мм) -> arc от устья.

    < 0 — расстояние от конца сосуда (`arc_tip`), >= 0 — от начала
    (`arc_root`). Ноль считается началом; ровно конец задавайте малым
    отрицательным (напр. -1e-6) — -0.0 намеренно НЕ различается (хрупко:
    int -0, JSON/YAML, abs/агрегации теряют знак).
    """
    offset = float(offset)
    if offset < 0.0:
        return arc_tip + offset
    return arc_root + offset


def _window_bounds(window, arc_root: float, arc_tip: float) -> tuple:
    """(lo, hi) дуги по окну (a, b) со знаковыми смещениями, отсортированные."""
    p0 = _offset_to_arc(window[0], arc_root, arc_tip)
    p1 = _offset_to_arc(window[1], arc_root, arc_tip)
    return (p0, p1) if p0 <= p1 else (p1, p0)


def pick_start_end(rng, rooted: Rooted, label: int, strict: bool = False) -> dict | None:
    """Случайные start (окно [A,B]) и end (потомок, окно [C,D]).

    Окна — знаковые смещения в мм: >= 0 от начала сосуда, < 0 от конца (см.
    `WINDOWS`). Если окна не влезают в короткий сосуд, они ужимаются так, чтобы
    всё равно получить фрагмент длины >= MIN_FRAGMENT_MM (флаг clamped). При
    strict=True такие сосуды пропускаются.
    """
    idx, arc = vessel_nodes(rooted, label)
    if len(idx) < 2:
        return None
    name = SEGMENT_NAMES[label].upper()
    A, B, C, D = WINDOWS.get(name, (0.0, 5.0, 35.0, 40.0))

    arc_root, arc_tip = float(arc.min()), float(arc.max())
    length = arc_tip - arc_root
    if length < MIN_FRAGMENT_MM:
        return None

    # Окно старта, ужатое до [arc_root, arc_tip - MIN]. При strict=True такие
    # сосуды (короче окна, напр. ранняя бифуркация RCA) будут пропущены.
    s0, s1 = _window_bounds((A, B), arc_root, arc_tip)
    lo = max(s0, arc_root)
    hi = min(s1, arc_tip - MIN_FRAGMENT_MM)
    clamped = (lo > s0 + 1e-9) or (hi < s1 - 1e-9)
    if hi < lo:
        lo = hi = arc_root
        clamped = True
    start_arc = lo if hi - lo < 1e-9 else rng.uniform(lo, hi)
    start = int(idx[int(np.argmin(np.abs(arc - start_arc)))])

    # Конец ищем ТОЛЬКО среди потомков старта: это гарантирует, что путь идёт
    # вдоль одного сосуда, а не «вверх к развилке и вниз в другую ветвь».
    desc = rooted.descendants(start)
    cand = np.array([int(i) for i in idx if int(i) in desc])
    if len(cand) == 0:
        return None
    arc_cand = np.array([rooted.arc[int(i)] for i in cand])

    # Окно конца по знаковым смещениям, ужатое до [start + MIN, arc_tip].
    e0, e1 = _window_bounds((C, D), arc_root, arc_tip)
    raw_end_arc = e0 if e1 - e0 < 1e-9 else rng.uniform(e0, e1)
    end_arc = float(np.clip(raw_end_arc, rooted.arc[start] + MIN_FRAGMENT_MM, arc_tip))
    clamped = clamped or abs(end_arc - raw_end_arc) > 1e-9
    end = int(cand[int(np.argmin(np.abs(arc_cand - end_arc)))])

    if rooted.arc[end] - rooted.arc[start] < MIN_FRAGMENT_MM - 1e-9:
        return None
    if strict and clamped:
        return None
    return {"start": start, "end": end,
            "arc_start": float(rooted.arc[start]), "arc_end": float(rooted.arc[end]),
            "windows_mm": [A, B, C, D], "clamped": bool(clamped)}


# --------------------------------------------------------------------------- #
#  Клик и фрагмент                                                            #
# --------------------------------------------------------------------------- #

def make_click(rng, vm, mask_bin, affine, point: np.ndarray, tangent: np.ndarray,
               zooms: tuple, radius: float, region_cov: float = CLICK_REGION_COV,
               depth_frac: float = CLICK_DEPTH_FRAC) -> dict:
    """Смещение клика: случайный воксель просвета в плоскости сечения точки.

    Слой вдоль касательной — как у радиуса и QC-среза (полвокселя). Просвет в
    плоскости задаём как `cov >= region_cov` (0.5) и оставляем кандидатов не
    ближе `margin` к его границе — `depth = distance_transform_edt(region)` в мм.
    `margin = max(CLICK_DEPTH_MIN_VOX·min(zooms), depth_frac·radius)`. Если
    подходящих нет — точка остаётся на центрлинии (`inside=False`).
    """
    half_slab = 0.5 * float(min(zooms))
    rel = vm.points - point
    axial = rel @ tangent
    lateral = np.linalg.norm(rel - np.outer(axial, tangent), axis=1)
    ok = (np.abs(axial) <= half_slab) & (lateral <= float(radius))
    if not ok.any():
        return {"click": point, "offset_mm": 0.0, "inside": False}

    cand = rel[ok]
    cov, e1, e2, step = section_coverage(mask_bin, affine, point, tangent,
                                         half_mm=SECTION_HALF_MM)
    n = cov.shape[0]
    iu = np.rint((cand @ e1 + SECTION_HALF_MM) / step).astype(int)
    iv = np.rint((cand @ e2 + SECTION_HALF_MM) / step).astype(int)
    inside = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
    # Глубина от границы просвета в плоскости (region = cov >= region_cov), мм.
    depth = distance_transform_edt(cov >= region_cov, sampling=(step, step))
    dv = np.zeros(len(cand))
    dv[inside] = depth[iv[inside], iu[inside]]
    margin = max(CLICK_DEPTH_MIN_VOX * float(min(zooms)), depth_frac * float(radius))
    good = dv > margin
    if not good.any():
        return {"click": point, "offset_mm": 0.0, "inside": False}
    click = point + cand[good][int(rng.integers(int(good.sum())))]
    return {"click": click, "offset_mm": float(np.linalg.norm(click - point)), "inside": True}


def extract_fragment(mask: np.ndarray, attr: Attribution, tree: str,
                     path_nodes, label: int) -> tuple[np.ndarray, dict]:
    """Воксели маски, приписанные точкам пути; приёмка по одной 26-компоненте.

    Фильтр по label обязателен: на развилке воксель соседнего сосуда может
    оказаться ближе к точке нашего пути и без фильтра попадёт во фрагмент.
    """
    nodes = attr.global_nodes(tree, np.asarray(path_nodes, dtype=int))
    take = np.isin(attr.node, nodes) & (attr.labels == label)
    vox = attr.voxels[take]
    binary = np.zeros(mask.shape, dtype=np.uint8)
    if len(vox):
        binary[vox[:, 0], vox[:, 1], vox[:, 2]] = 1
    comp, ncomp = cc_label(binary, structure=CONNECTIVITY_26)
    sizes = sorted(np.bincount(comp.ravel())[1:].tolist(), reverse=True)
    return binary, {"n_voxels": int(binary.sum()), "n_components": int(ncomp),
                    "component_sizes": sizes[:5]}


# --------------------------------------------------------------------------- #
#  QC-картинка                                                                #
# --------------------------------------------------------------------------- #

QC_CROP_HALF_MM = 4.0     # окно ПОКАЗА QC-среза (кроп из сетки SECTION_HALF_MM)


def _crop_view(cov, step, crop_half):
    """Центральный кроп сетки `cov` под окно ±crop_half мм (0 — центр сетки)."""
    n = cov.shape[0]
    c = (n - 1) // 2
    k = min(int(round(crop_half / step)), c)
    sub = cov[c - k:c + k + 1, c - k:c + k + 1]
    e = k * step
    return sub, [-e, e, -e, e]


def save_qc(mask, affine, zooms, tree, vessel_label, path_nodes, path_arc,
            path_radius, tang, clicks, frag, path_out) -> None:
    cl = tree.cl
    fig = plt.figure(figsize=(19, 5.4), dpi=130)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.2, 1.2], wspace=0.3)

    # Срез строится в плоскости по БИНАРНОЙ маске целевого сосуда (mask == label):
    # иначе map_coordinates интерполирует номера меток и граница/площадь врут.
    vessel_bin = (mask == vessel_label).astype(np.float32)
    for j, key in enumerate(("start", "end")):
        i = path_nodes[0] if key == "start" else path_nodes[-1]
        p = cl.points[i]
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor("#101014")
        cov, e1, e2, step = section_coverage(vessel_bin, affine, p, tang[i],
                                             half_mm=SECTION_HALF_MM)
        sub, ext = _crop_view(cov, step, QC_CROP_HALF_MM)
        ax.imshow(sub, extent=ext, origin="lower", cmap="Greens",
                  vmin=0.0, vmax=1.0, alpha=0.9)
        r = path_radius[0] if key == "start" else path_radius[-1]
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="cyan", ls="--", lw=1.6))
        ax.plot([0], [0], "o", color="white", ms=9, mec="black", zorder=6)
        click = clicks[j]["click"]
        du, dv = (click - p) @ e1, (click - p) @ e2
        ax.plot([du], [dv], "X", color="#FF2D95", ms=12, mec="black", zorder=7)
        ax.plot([0, du], [0, dv], color="#FF2D95", lw=2)
        ax.set_aspect("equal")
        ax.set_title(f"Сечение {j + 1}: r={r:.2f} мм, сдвиг {clicks[j]['offset_mm']:.2f} мм",
                     fontsize=10)
        ax.set_xlabel("ось 1, мм")
        ax.set_ylabel("ось 2, мм")
        ax.legend(handles=[
            mlines.Line2D([], [], marker="o", color="none", mec="black", mfc="white",
                          ls="none", ms=9, label="точка центрлинии"),
            mlines.Line2D([], [], marker="X", color="none", mec="black", mfc="#FF2D95",
                          ls="none", ms=11, label="клик"),
            mlines.Line2D([], [], color="cyan", ls="--", label="радиус по сечению")],
            fontsize=8, loc="upper right")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(path_arc, path_radius, color="#2D7DFF", lw=1.6)
    for a, r in ((path_arc[0], path_radius[0]), (path_arc[-1], path_radius[-1])):
        ax.plot([a], [r], "X", color="#FF2D95", ms=10, mec="black")
    ax.set_xlabel("расстояние от устья, мм")
    ax.set_ylabel("радиус, мм")
    ax.set_title(f"Радиус вдоль фрагмента ({len(path_nodes)} точек)", fontsize=10)

    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor("#101014")
    # MIP фрагмента вдоль y, в мм по спейсингу, обрезанный по bbox фрагмента:
    # залитая маска, а не россыпь точек, и без пустых полей вокруг.
    mip = frag.max(axis=1).T                     # (Z, X)
    vox = np.argwhere(frag > 0)
    pad = 5
    x0, x1 = int(vox[:, 0].min()) - pad, int(vox[:, 0].max()) + pad
    z0, z1 = int(vox[:, 2].min()) - pad, int(vox[:, 2].max()) + pad
    x0, x1 = max(0, x0), min(frag.shape[0] - 1, x1)
    z0, z1 = max(0, z0), min(frag.shape[2] - 1, z1)
    ext = [x0 * zooms[0], x1 * zooms[0], z0 * zooms[2], z1 * zooms[2]]
    ax.imshow(mip[z0:z1 + 1, x0:x1 + 1], origin="lower", extent=ext, cmap="Greens",
              aspect="equal", interpolation="nearest")
    ax.set_title(f"Маска фрагмента (MIP вдоль y): {int(frag.sum())} вокселей",
                 fontsize=10)
    ax.set_xlabel("x, мм")
    ax.set_ylabel("z, мм")
    ax.tick_params(labelsize=8)

    name = SEGMENT_NAMES.get(vessel_label, vessel_label)
    fig.suptitle(f"Скан {cl.scan_id} · {name} · длина {path_arc[-1] - path_arc[0]:.1f} мм",
                 fontsize=13)
    path_out = common._ensure_parent(path_out)
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Генерация одного сосуда                                                    #
# --------------------------------------------------------------------------- #

def _json_point(i, rooted, path_arc, path_radius, path_edt, path_kaw, clicks, j) -> dict:
    p = rooted.cl.points[i]
    k = -1 if j == 1 else 0
    return {
        "dist_from_ostium_mm": round(float(path_arc[k]), 3),
        "line_xyz": [round(float(v), 4) for v in p],
        "click_xyz": [round(float(v), 4) for v in clicks[j]["click"]],
        "click_offset_mm": round(float(clicks[j]["offset_mm"]), 4),
        "click_inside": bool(clicks[j]["inside"]),
        "radius_mm": round(float(path_radius[k]), 4),
        "radius_kawaleri_mm": round(float(path_kaw[k]), 4),
        "radius_edt_mm": round(float(path_edt[k]), 4),
    }


def generate_vessel(scan_id, mask, affine, zooms, centered: dict, attr: Attribution,
                    radii: dict, vms: dict, label: int, rng, out_dir, index_rows,
                    strict: bool = False) -> bool:
    tree_name = "left" if label in (2, 3) else "right"
    rooted = centered[tree_name]
    cl = rooted.cl

    prof = radii[tree_name]
    tang = prof["tangent"]
    vm = vms[label]
    # Бинарная маска целевого сосуда — один раз на сосуд (для срезов/радиуса).
    vessel_bin = (mask == label).astype(np.float32)

    # Чистим возможные старые файлы этого сосуда (важно при смене флагов).
    sample_id = f"{scan_id}_{SEGMENT_NAMES[label].lower()}"
    for stale in (out_dir / "fragments" / f"{sample_id}.nii.gz",
                  out_dir / "samples" / f"{sample_id}.json",
                  out_dir / "qc" / f"{sample_id}.png"):
        stale.unlink(missing_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        pick = pick_start_end(rng, rooted, label, strict)
        if pick is None:
            continue
        path_nodes = path_between(rooted, pick["start"], pick["end"])
        path_arc = np.array([rooted.arc[i] for i in path_nodes])
        if path_arc[-1] - path_arc[0] < MIN_FRAGMENT_MM:
            continue

        frag, stats = extract_fragment(mask, attr, tree_name, path_nodes, label)
        if stats["n_voxels"] == 0 or stats["n_components"] != 1:
            continue

        # Новый радиус/площадь: интеграл покрытия по плоскости среза.
        r_new = np.sqrt(np.array([
            section_area(vessel_bin, affine, cl.points[i], tang[i]) / np.pi
            for i in path_nodes]))
        # Старый метод (Кавалери по 3D-вокселям) — для сравнения.
        r_kaw = prof["radius_section"][path_nodes]
        r_edt = prof["radius_edt"][path_nodes]
        # Сглаживание профиля вдоль фрагмента: маска неровная, сырой r шумит.
        r_plot = median_filter(r_new, size=5, mode="nearest") if len(r_new) >= 5 else r_new
        clicks = [
            make_click(rng, vm, vessel_bin, affine, cl.points[path_nodes[0]],
                       tang[path_nodes[0]], zooms, float(r_new[0])),
            make_click(rng, vm, vessel_bin, affine, cl.points[path_nodes[-1]],
                       tang[path_nodes[-1]], zooms, float(r_new[-1])),
        ]

        name = SEGMENT_NAMES[label]
        nii = out_dir / "fragments" / f"{sample_id}.nii.gz"
        save_mask_nifti(frag, affine, zooms, nii)

        sample = {
            "scan_id": scan_id, "sample_id": sample_id, "vessel": name, "label": label,
            "tree": tree_name, "attempt": attempt,
            "windows_mm": pick["windows_mm"], "windows_clamped": pick["clamped"],
            "n_points": len(path_nodes),
            "length_mm": round(float(path_arc[-1] - path_arc[0]), 3),
            "points": [_json_point(i, rooted, path_arc, r_plot, r_edt, r_kaw, clicks, j)
                       for j, i in enumerate((path_nodes[0], path_nodes[-1]))],
            "centerline": [
                {"arc_mm": round(float(a), 4),
                 "xyz": [round(float(v), 4) for v in cl.points[i]],
                 "radius_mm": round(float(r), 4),
                 "radius_raw_mm": round(float(rr), 4),
                 "radius_kawaleri_mm": round(float(kk), 4)}
                for i, a, r, rr, kk in zip(path_nodes, path_arc, r_plot, r_new, r_kaw)],
            "fragment": stats,
            "mask_nii": str(nii),
        }
        common._ensure_parent(out_dir / "samples" / f"{sample_id}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")

        save_qc(mask, affine, zooms, rooted, label, path_nodes, path_arc, r_plot,
                tang, clicks, frag, out_dir / "qc" / f"{sample_id}.png")
        index_rows.append({
            "sample_id": sample_id, "scan_id": scan_id, "vessel": name,
            "length_mm": sample["length_mm"], "n_voxels": stats["n_voxels"],
            "mask_nii": str(nii),
        })
        print(f"  {sample_id}: длина {sample['length_mm']:.1f} мм, "
              f"вокселей {stats['n_voxels']}, сдвиги "
              f"{clicks[0]['offset_mm']:.2f}/{clicks[1]['offset_mm']:.2f} мм, попытка {attempt}")
        return True
    print(f"  {scan_id} {SEGMENT_NAMES[label]}: не удалось собрать фрагмент")
    return False


def main() -> int:
    args = common.parse_common_args("Генерация фрагментов: 3 на скан")
    scan_ids = common.resolve_scan_ids(args)
    print(f"сканов: {len(scan_ids)} (split {args.split})")
    out_dir = args.out / "generate"
    for sub in ("fragments", "samples", "radii", "qc"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    for scan_id in tqdm(scan_ids, desc="generate", unit="scan", mininterval=5.0):
        ps = prepare_scan(scan_id, use_cache=not args.no_cache)
        # Отдельно сохраняем радиус в каждой точке дерева — это самостоятельный
        # результат, полезный и вне фрагментов.
        for t in ("left", "right"):
            np.savez_compressed(out_dir / "radii" / f"{scan_id}_{t}.npz", **ps.radii[t])

        for label in TARGET_IDS:
            tree = "left" if label in (2, 3) else "right"
            if label not in ps.vms or not np.any(ps.lines[tree].labels == label):
                continue
            rng = common.make_rng(common.SEED, scan_id, label)
            generate_vessel(scan_id, ps.mask, ps.affine, ps.zooms, ps.centered, ps.attr,
                            ps.radii, ps.vms, label, rng, out_dir, index_rows,
                            strict=args.strict_windows)

    points = out_dir / "points.jsonl"
    points.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index_rows),
                      encoding="utf-8")
    print(f"\nготово: {out_dir} ({len(index_rows)} примеров)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
