"""02 — наложение цветной маски сосудов и центрлиний на КТ для ImageJ.

Пишет:
  out/overlay/<scan>_overlay_rgb.tif   — RGB multi-page TIFF (основной формат);
  out/overlay/<scan>_labels.tif        — скалярные метки 1..14 (TIFF);
  out/overlay/<scan>_overlay_rgb.nii.gz, <scan>_labels.nii.gz — NIfTI-варианты
      (RGB-NIfTI ImageJ/Fiji читают плохо — соляризация/битые каналы);
  out/overlay/<scan>_lut.txt           — id, имя, RGB, число вокселей;
  out/overlay/<scan>_overlay.png       — 2 строки × 3 анатомические проекции:
      верх — сосуды (tube-MIP без разметки), низ — маски и центрлинии;
  out/overlay/README.txt               — рецепт просмотра в ImageJ/Fiji.

Серая основа — КТ в окне HU (`HU_WINDOW`, гамма `GREY_GAMMA`); серый MIP строится
в приоритете по трубке вокруг центрлиний, поверх — полупрозрачная цветная маска и
чёрные центрлинии.

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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import common
from common import (PALETTE, PROJECTION_PANELS, SEGMENT_NAMES, TUBE_K, TUBE_SOFT_MM,
                    check_geometry, grey_from_hu, mask_to_rgb, orient_mip, paint_points,
                    read_centerline, read_ct, read_mask, save_rgb_nifti,
                    save_mask_nifti, save_rgb_tiff, save_labels_tiff,
                    tube_weight, view_limits, voxel_in_view)

ALPHA = 0.55           # доля цвета маски в смешении
LINE_COLOR = (0, 0, 0)         # чёрные центрлинии: видны на ярких сосудах/костях
LINE_RADIUS_VOX = 0    # без дилатации — линия толщиной 1 воксель, тоньше
COVER_SIGMA_PX = 0.0   # размытие cover гауссом, пикселей (0 = выключено)
# TUBE_K / TUBE_SOFT_MM (параметры MIP-приоритета по трубке) — в common.py.


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


def save_png(ct, mask, affine, zooms, lines, path) -> None:
    """2 строки × 3 анатомические проекции (MIP).

    Верхняя строка — сосуды (tube-MIP без разметки), нижняя — маски и центрлинии.
    Основные оси — воксельные индексы; aspect = spacing_row/spacing_col, поэтому
    1 мм по вертикали равен 1 мм по горизонтали (z=0.5 против xy=0.357). Вид
    обрезается по bbox маски с запасом. Сверху/справа — дублирующие оси в мм.
    Центрлинии рисуем ПО ЯЧЕЙКАМ; иначе линия соединяет точки в порядке массива
    и даёт «спагетти»-пятно.

    Серый MIP строится в приоритете по трубке: на пикселях, накрытых проекцией
    трубки вокруг центрлиний, берётся MIP только внутри трубки (целевой сосуд не
    перекрывается чужой кровью), иначе — обычный MIP (другие сосуды видны).
    """
    grey = grey_from_hu(ct)
    tube = tube_weight(mask, affine, zooms, lines, TUBE_K, TUBE_SOFT_MM)   # мягкий вес трубки
    grey_tube = grey * tube
    # Всё, что не зависит от панели, считаем ОДИН раз (сейчас mask_to_rgb — дорогой).
    tint_all = mask_to_rgb(mask)                       # uint8 (X,Y,Z,3)
    mask_bin = (mask > 0).astype(np.float32)           # для по-пиксельной альфы
    present = [int(v) for v in np.unique(mask) if v]
    view = view_limits(mask, affine)
    # Две строки: сверху — только сосуды (tube-MIP, без разметки), снизу — разметка.
    # constrained_layout сам резервирует место под suptitle и fig.legend (в отличие
    # от tight_layout, который suptitle игнорирует). Легенду ставим через
    # loc="outside lower center" — тогда движок выделяет ей отдельную полосу снизу
    # и она не налезает на подписи оси x второй строки.
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.2), dpi=130, layout="constrained")
    for k, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
        sr, sc = float(zooms[row]), float(zooms[col])
        base, flip_r, flip_c = orient_mip(grey, drop, row, col, affine)
        cover, _, _ = orient_mip(tube, drop, row, col, affine)
        # Гаусс по cover — опционально (COVER_SIGMA_PX>0); мягкость края задаётся
        # в основном TUBE_SOFT_MM в 3D, что согласованно влияет и на target.
        if COVER_SIGMA_PX > 0:
            cover = gaussian_filter(cover, sigma=COVER_SIGMA_PX, mode="nearest")
        target, _, _ = orient_mip(grey_tube, drop, row, col, affine)
        # Приоритет: где трубка перекрывает луч, показываем только её содержимое.
        proj = np.clip(np.maximum(base * (1.0 - cover), target), 0.0, 1.0)
        tint, _, _ = orient_mip(tint_all, drop, row, col, affine)
        mask_any, _, _ = orient_mip(mask_bin, drop, row, col, affine)

        for j in (0, 1):                   # j=0 — сосуды; j=1 — разметка
            ax = axes[j, k]
            # aspect задаём у imshow: без него сбрасывается в equal.
            ax.imshow(proj, cmap="gray", origin="lower", interpolation="nearest",
                      vmin=0.0, vmax=1.0, aspect=sr / sc)
            ax.set_xlim(*view[col])
            ax.set_ylim(*view[row])
            ax.set_xlabel(f"{clab}, индекс")
            ax.set_ylabel(f"{rlab}, индекс")
            ax.tick_params(labelsize=8)
            secx = ax.secondary_xaxis("top",
                                      functions=(lambda i: i * sc, lambda m: m / sc))
            secy = ax.secondary_yaxis("right",
                                      functions=(lambda i: i * sr, lambda m: m / sr))
            secx.set_xlabel(f"{clab}, мм", fontsize=9)
            secy.set_ylabel(f"{rlab}, мм", fontsize=9)
            secx.tick_params(labelsize=7)
            secy.tick_params(labelsize=7)

        axes[0, k].set_title(f"{title} · сосуды", fontsize=11)
        axes[1, k].set_title("разметка: маски + центрлинии", fontsize=11)
        # Маска и центрлинии — только в нижней строке.
        axes[1, k].imshow(tint, origin="lower", interpolation="nearest",
                          alpha=ALPHA * mask_any, aspect=sr / sc)
        for cl in lines.values():
            c, r = voxel_in_view(cl.points, affine, row, col, proj.shape, flip_r, flip_c)
            for cell in cl.cells:          # ячейка = непрерывный путь вдоль сосуда
                axes[1, k].plot(c[cell], r[cell], "-", color="black", lw=1.1, zorder=6)

    # Легенда: цвет маски, смешанный с БЕЛЫМ фоном (как фон легенды).
    badges = [(1 - ALPHA) * 1.0 + ALPHA * np.array(PALETTE[l]) / 255.0 for l in present]
    handles = [Patch(facecolor=badges[i], edgecolor="black",
                     label=f"{l} {SEGMENT_NAMES.get(l, l)}")
               for i, l in enumerate(present)]
    handles.append(Line2D([0], [0], color="black", lw=1.4, label="центральная линия"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 7),
               frameon=False, fontsize=9)
    fig.suptitle("Верх — сосуды (tube-MIP, без разметки); низ — маски и центрлинии "
                 f"(цвет маски смешан с КТ, α={ALPHA:g})", fontsize=13)
    path = common._ensure_parent(path)
    fig.savefig(path)
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
    for scan_id in tqdm(scan_ids, desc="overlay", unit="scan", mininterval=5.0):
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
    save_readme(out_dir / "README.txt")
    print(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
