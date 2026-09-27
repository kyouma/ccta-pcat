"""02 — наложение цветной маски сосудов и центрлиний на КТ для ImageJ.

Пишет:
  out/overlay/<scan>_overlay_rgb.tif   — RGB multi-page TIFF (основной формат);
  out/overlay/<scan>_labels.tif        — скалярные метки 1..14 (TIFF);
  out/overlay/<scan>_overlay_rgb.nii.gz, <scan>_labels.nii.gz — NIfTI-варианты
      (RGB-NIfTI ImageJ/Fiji читают плохо — соляризация/битые каналы);
  out/overlay/<scan>_lut.txt           — id, имя, RGB, число вокселей;
  out/overlay/<scan>_overlay.png       — три анатомические проекции для QC;
  out/overlay/README.txt               — рецепт просмотра в ImageJ/Fiji.

Серая основа — КТ в окне HU с гамма-коррекцией, поверх — полупрозрачная цветная
маска и чёрные центрлинии.

Запуск:
    python 02_overlay.py --ids 961
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
from common import (PALETTE, SEGMENT_NAMES, read_centerline, read_ct, read_mask,
                    check_geometry, grey_from_hu, mask_to_rgb, paint_points,
                    save_rgb_nifti, save_mask_nifti, save_rgb_tiff, save_labels_tiff,
                    to_voxel)

ALPHA = 0.55           # доля цвета маски в смешении
LINE_COLOR = (0, 0, 0)         # чёрные центрлинии: видны на ярких сосудах/костях
LINE_RADIUS_VOX = 0    # без дилатации — линия толщиной 1 воксель, тоньше


def build_overlay(ct, mask, affine, lines) -> np.ndarray:
    """RGB: серая основа КТ + полупрозрачная цветная маска + чёрные центрлинии.

    Строим uint8-объём (3 канала) без больших float-массивов на весь скана:
    сначала серый КТ, затем подмешиваем цвет только там, где есть маска.
    """
    grey = grey_from_hu(ct)                       # float32 [0,1], (X,Y,Z)
    rgb = np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=3)

    # Полупрозрачное смешение только в вокселях маски: out = (1-a)*grey + a*color.
    present = [int(v) for v in np.unique(mask) if v]
    for label in present:
        m = mask == label
        base = (1.0 - ALPHA) * grey[m][:, None] + ALPHA * np.array(PALETTE[label]) / 255.0
        rgb[m] = (np.clip(base, 0, 1) * 255).astype(np.uint8)

    # Центрлинии рисуем поверх маски и КТ, чтобы не терялись в смешении.
    points = np.vstack([cl.points for cl in lines.values()])
    line_vol = paint_points(mask.shape, affine, points, LINE_RADIUS_VOX)
    rgb[line_vol] = np.array(LINE_COLOR, dtype=np.uint8)
    return rgb


# Панели: (заголовок, ось проекции, вертикальная ось, горизонтальная ось,
# подпись вертикали, подпись горизонтали). Ориентация — анатомическая:
# сагиттальная/корональная с вертикалью S/I (z), аксиальная с вертикалью A/P (y).
PANELS = (
    ("Сагиттальная (вид сбоку, вдоль x)", 0, 2, 1, "S/I (z)", "A/P (y)"),
    ("Корональная (вид спереди, вдоль y)", 1, 2, 0, "S/I (z)", "R/L (x)"),
    ("Аксиальная (вид сверху, вдоль z)", 2, 1, 0, "A/P (y)", "R/L (x)"),
)


def _orient(volume, drop: int, row: int, col: int, affine) -> tuple:
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


def _voxel_in_view(points, affine, row, col, shape, flip_r, flip_c):
    """Мировые точки -> координаты (col, row) в развёрнутом изображении."""
    vox = to_voxel(affine, points)
    c = vox[:, col].copy()
    r = vox[:, row].copy()
    if flip_r:
        r = (shape[0] - 1) - r
    if flip_c:
        c = (shape[1] - 1) - c
    return c, r


def _view_limits(mask, affine, margin_mm: float = 25.0) -> dict:
    """Границы просмотра по каждой оси: bbox маски + запас, с учётом флипов.

    Показываем область сердца, а не весь кадр: так и скан «крупнее», и ярче.
    """
    idx = np.argwhere(mask > 0)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    n = np.array(mask.shape)
    spacing = np.abs(np.diag(affine))[:3]
    limits = {}
    for a in range(3):
        m = int(np.ceil(margin_mm / spacing[a])) if spacing[a] > 0 else 0
        a0, a1 = max(0, int(lo[a]) - m), min(n[a] - 1, int(hi[a]) + m)
        limits[a] = ((n[a] - 1 - a1, n[a] - 1 - a0) if affine[a, a] < 0 else (a0, a1))
    return limits


def save_png(ct, mask, affine, zooms, lines, path) -> None:
    """Три анатомические MIP: серая основа + цветная маска + центрлинии.

    Основные оси — воксельные индексы; aspect = spacing_row/spacing_col, поэтому
    1 мм по вертикали равен 1 мм по горизонтали (z=0.5 против xy=0.357). Вид
    обрезается по bbox маски с запасом. Сверху/справа — дублирующие оси в мм.
    Центрлинии рисуем ПО ЯЧЕЙКАМ; иначе линия соединяет точки в порядке массива
    и даёт «спагетти»-пятно.
    """
    grey = grey_from_hu(ct)
    present = [int(v) for v in np.unique(mask) if v]
    view = _view_limits(mask, affine)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), dpi=130)
    for k, (title, drop, row, col, rlab, clab) in enumerate(PANELS):
        ax = axes[k]
        sr, sc = float(zooms[row]), float(zooms[col])
        proj, flip_r, flip_c = _orient(grey, drop, row, col, affine)
        # aspect задаём у ОБОИХ imshow: второй вызов без aspect сбросил бы его.
        ax.imshow(proj, cmap="gray", origin="lower", interpolation="nearest",
                  vmin=0.0, vmax=1.0, aspect=sr / sc)
        tint, _, _ = _orient(mask_to_rgb(mask).astype(np.float32) / 255.0,
                             drop, row, col, affine)
        ax.imshow(tint, origin="lower", interpolation="nearest", alpha=ALPHA,
                  aspect=sr / sc)
        for cl in lines.values():
            c, r = _voxel_in_view(cl.points, affine, row, col, proj.shape, flip_r, flip_c)
            for cell in cl.cells:          # ячейка = непрерывный путь вдоль сосуда
                ax.plot(c[cell], r[cell], "-", color="black", lw=1.1, zorder=6)
        ax.set_xlim(*view[col])
        ax.set_ylim(*view[row])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"{clab}, индекс")
        ax.set_ylabel(f"{rlab}, индекс")
        ax.tick_params(labelsize=8)
        # Дублирующие оси в мм (index*spacing), сверху и справа.
        secx = ax.secondary_xaxis("top", functions=(lambda i: i * sc, lambda m: m / sc))
        secy = ax.secondary_yaxis("right", functions=(lambda i: i * sr, lambda m: m / sr))
        secx.set_xlabel(f"{clab}, мм", fontsize=9)
        secy.set_ylabel(f"{rlab}, мм", fontsize=9)
        secx.tick_params(labelsize=7)
        secy.tick_params(labelsize=7)

    # В легенде показываем уже СМЕШАННЫЙ с серой основой цвет (как на графике),
    # а не чистую палитру: берём представительный средний тон фона.
    bg = 0.4
    handles = [Patch(facecolor=(1 - ALPHA) * bg + ALPHA * np.array(PALETTE[l]) / 255.0,
                     edgecolor="black", label=f"{l} {SEGMENT_NAMES.get(l, l)}")
               for l in present]
    handles.append(Line2D([0], [0], color="black", lw=1.4, label="центральная линия"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
               frameon=False, fontsize=9)
    fig.suptitle("Маска сосудов и центрлинии на КТ "
                 f"(цвет маски смешан с КТ, α={ALPHA:g})", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = common._ensure_parent(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_lut(mask, path) -> None:
    path = common._ensure_parent(path)
    present = [int(v) for v in np.unique(mask) if v]
    with path.open("w", encoding="utf-8") as f:
        f.write("id\tname\tR\tG\tB\tvoxels\n")
        for v in present:
            r, g, b = PALETTE.get(v, (255, 255, 255))
            f.write(f"{v}\t{SEGMENT_NAMES.get(v, v)}\t{r}\t{g}\t{b}\t"
                    f"{int((mask == v).sum())}\n")


def save_readme(path) -> None:
    path = common._ensure_parent(path)
    path.write_text(
        "Просмотр в ImageJ / Fiji:\n"
        "1. РЕКОМЕНДУЕТСЯ: File > Open <scan>_overlay_rgb.tif (RGB multi-page\n"
        "   TIFF, ImageJ сам собирает RGB-гиперстек, калибровка x/y и z из тегов).\n"
        "2. Для анализа меток: <scan>_labels.tif (или _labels.nii.gz) + цвета\n"
        "   из <scan>_lut.txt.\n"
        "3. RGB NIfTI <scan>_overlay_rgb.nii.gz — для инструментов, которые его\n"
        "   умеют; ImageJ/Fiji читают RGB-NIfTI некорректно.\n"
        "4. 3D: Plugins > 3D Viewer, или Image > Stacks > 3D Project.\n",
        encoding="utf-8")


def main() -> int:
    args = common.parse_common_args("Наложение маски сосудов на КТ")
    scan_ids = common.resolve_scan_ids(args)
    print(f"сканов: {len(scan_ids)} (split {args.split})")

    out_dir = args.out / "overlay"
    for scan_id in scan_ids:
        ct, affine = read_ct(scan_id)
        mask, affine_m, zooms = read_mask(scan_id)
        check_geometry(affine, affine_m)
        lines = {t: read_centerline(scan_id, t) for t in ("left", "right")}

        rgb = build_overlay(ct, mask, affine, lines)
        save_rgb_nifti(rgb, affine, zooms, out_dir / f"{scan_id}_overlay_rgb.nii.gz")
        save_mask_nifti(mask, affine, zooms, out_dir / f"{scan_id}_labels.nii.gz")
        save_rgb_tiff(rgb, zooms, out_dir / f"{scan_id}_overlay_rgb.tif")
        save_labels_tiff(mask, zooms, out_dir / f"{scan_id}_labels.tif")
        save_lut(mask, out_dir / f"{scan_id}_lut.txt")
        save_png(ct, mask, affine, zooms, lines, out_dir / f"{scan_id}_overlay.png")
        print(f"  {scan_id}: NIfTI + TIFF + labels + LUT + PNG")
    save_readme(out_dir / "README.txt")
    print(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
