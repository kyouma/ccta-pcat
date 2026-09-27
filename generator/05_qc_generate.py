"""05 — анатомический QC сгенерированных фрагментов (в духе overlay PNG).

Для каждого скана и каждого сосуда, для которого уже есть пример из
`04_generate.py`, строит контекстную картинку 2×3 по анатомическим проекциям:
  верх — КТ + все маски сосудов (полупрозрачно) + фрагмент + центрлиния + клики;
  низ  — только фрагмент (что именно уходит в пример) + центрлиния + клики.

Это дополняет per-sample QC из 04 (сечения/радиус/MIP): сразу видно, что
фрагмент сидит на нужном сосуде и не зацепил соседний.

Запуск:
    python 05_qc_generate.py --ids 69
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

import common
from common import (PROJECTION_PANELS, SEGMENT_NAMES, TARGET_IDS, TUBE_K, TUBE_SOFT_MM,
                    check_geometry, grey_from_hu, mask_to_rgb, orient_mip, read_ct,
                    view_limits, voxel_in_view, tube_weight)

ALPHA = 0.45            # прозрачность масок сосудов в верхней строке
FRAG_COLOR = (1.0, 0.85, 0.0)   # фрагмент — ярко-жёлтый
CLICK_START = "#FF2D95"         # клик старта
CLICK_END = "#00E5FF"           # клик конца


def load_sample(out_dir, scan_id: str, label: int):
    """Читает JSON примера и маску фрагмента; None, если примера нет."""
    sid = f"{scan_id}_{SEGMENT_NAMES[label].lower()}"
    sample_path = out_dir / "samples" / f"{sid}.json"
    frag_path = out_dir / "fragments" / f"{sid}.nii.gz"
    if not (sample_path.exists() and frag_path.exists()):
        return None
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    frag = np.asanyarray(nib.load(str(frag_path)).dataobj).astype(np.float32)
    return sample, frag


def build_context(ct, mask, affine, zooms, lines):
    """Тяжёлые объёмы, не зависящие от сосуда: серый КТ и приоритет по трубке."""
    grey = grey_from_hu(ct)
    tube = tube_weight(mask, affine, zooms, lines, TUBE_K, TUBE_SOFT_MM)
    return {"grey": grey, "grey_tube": grey * tube, "tube": tube}


def save_context_png(ctx, mask, affine, zooms, sample, frag, path) -> None:
    grey, tube, grey_tube = ctx["grey"], ctx["tube"], ctx["grey_tube"]
    tint_all = mask_to_rgb(mask)                       # uint8 (X,Y,Z,3)
    mask_bin = (mask > 0).astype(np.float32)
    frag_bin = (frag > 0).astype(np.float32)
    view = view_limits(mask, affine)

    cl_xyz = np.array([p["xyz"] for p in sample["centerline"]], float)
    clicks = np.array([sample["points"][0]["click_xyz"],
                       sample["points"][1]["click_xyz"]], float)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11.2), dpi=130, layout="constrained")
    for k, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
        sr, sc = float(zooms[row]), float(zooms[col])
        base, flip_r, flip_c = orient_mip(grey, drop, row, col, affine)
        cover, _, _ = orient_mip(tube, drop, row, col, affine)
        target, _, _ = orient_mip(grey_tube, drop, row, col, affine)
        # Приоритет по трубке: где трубка перекрывает луч, показываем только её.
        proj = np.clip(np.maximum(base * (1.0 - cover), target), 0.0, 1.0)
        tint, _, _ = orient_mip(tint_all, drop, row, col, affine)
        mask_any, _, _ = orient_mip(mask_bin, drop, row, col, affine)
        frag_proj, _, _ = orient_mip(frag_bin, drop, row, col, affine)

        for j, show_masks in enumerate((True, False)):
            ax = axes[j, k]
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

            if show_masks:                 # все сосуды — приглушённо, только сверху
                ax.imshow(tint, origin="lower", interpolation="nearest",
                          alpha=ALPHA * mask_any, aspect=sr / sc)
            # Фрагмент — ярким поверх всего.
            rgba = np.zeros((*frag_proj.shape, 4))
            rgba[frag_proj > 0] = (*FRAG_COLOR, 0.85)
            ax.imshow(rgba, origin="lower", interpolation="nearest", aspect=sr / sc)

            # Центрлиния фрагмента (путь упорядочен вдоль сосуда — рисуем линией).
            c, r = voxel_in_view(cl_xyz, affine, row, col, proj.shape, flip_r, flip_c)
            ax.plot(c, r, "-", color="black", lw=1.6, zorder=6)
            # Точки клика.
            cc, rr = voxel_in_view(clicks, affine, row, col, proj.shape, flip_r, flip_c)
            ax.plot(cc[0], rr[0], "X", color=CLICK_START, ms=13, mec="black",
                    mew=1.2, zorder=7)
            ax.plot(cc[1], rr[1], "X", color=CLICK_END, ms=13, mec="black",
                    mew=1.2, zorder=7)

        axes[0, k].set_title(f"{title} · маски + фрагмент", fontsize=11)
        axes[1, k].set_title("только фрагмент", fontsize=11)

    handles = [
        Patch(facecolor=FRAG_COLOR, edgecolor="black", label="фрагмент сосуда"),
        Line2D([0], [0], color="black", lw=1.6, label="центрлиния фрагмента"),
        Line2D([0], [0], marker="X", color="none", mec="black", mfc=CLICK_START,
               ls="none", ms=11, label="клик старта"),
        Line2D([0], [0], marker="X", color="none", mec="black", mfc=CLICK_END,
               ls="none", ms=11, label="клик конца"),
        Patch(facecolor="0.6", edgecolor="black", label="маски сосудов (верх)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=9)
    p0, p1 = sample["points"]
    fig.suptitle(
        f"Скан {sample['scan_id']} · {sample['vessel']} · длина {sample['length_mm']:.1f} мм"
        f" · вокселей {sample['fragment']['n_voxels']}"
        f" · сдвиги {p0['click_offset_mm']:.2f}/{p1['click_offset_mm']:.2f} мм",
        fontsize=13)
    path = common._ensure_parent(path)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = common.parse_common_args("Анатомический QC сгенерированных фрагментов")
    scan_ids = common.resolve_scan_ids(args)
    print(f"сканов: {len(scan_ids)} (split {args.split})")

    out_dir = args.out / "generate"
    if not (out_dir / "samples").exists():
        print(f"нет примеров в {out_dir / 'samples'} — сначала запусти 04_generate.py")
        return 1

    n_made = 0
    for scan_id in tqdm(scan_ids, desc="qc_generate", unit="scan", mininterval=5.0):
        # Сначала проверяем, есть ли вообще примеры у скана: если нет — не тратим
        # время на чтение КТ и дорогой tube_weight.
        samples = []
        for label in TARGET_IDS:
            loaded = load_sample(out_dir, scan_id, label)
            if loaded is not None:
                samples.append((label, *loaded))
        if not samples:
            continue
        ps = common.prepare_scan(scan_id, use_cache=not args.no_cache)
        ct, affine = read_ct(scan_id)
        check_geometry(affine, ps.affine)
        ctx = build_context(ct, ps.mask, ps.affine, ps.zooms, ps.lines)
        for label, sample, frag in samples:
            sid = f"{scan_id}_{SEGMENT_NAMES[label].lower()}"
            save_context_png(ctx, ps.mask, ps.affine, ps.zooms, sample, frag,
                             out_dir / "qc" / f"{sid}_context.png")
            n_made += 1
    print(f"\nготово: {n_made} контекстных QC в {out_dir / 'qc'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
