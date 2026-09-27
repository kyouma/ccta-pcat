"""03 — QC и анализ данных для выбора параметров.

Проверяет и собирает статистику:
  * совпадение геометрии КТ и маски;
  * центрлиния внутри маски, согласие метки точки и ближайшего вокселя;
  * snap_mm (расстояние от точки центрлинии до ближайшего вокселя маски);
  * целостность дерева: компоненты связности, недостижимые из устья точки;
  * радиус сосуда по сечению против EDT (cross-check, как в ImageCAS-X)
    для целевых сосудов LAD / LCx / RCA;
  * длины ветвей (arc от устья до дистального конца) целевых сосудов.

Пишет текстовый отчёт, CSV по сканам/сосудам и PNG-гистограммы.

Запуск:
    python 03_qc.py --split test --limit 20
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import common
from common import (MAX_SNAP_MM, RADIUS_SANITY, SEGMENT_NAMES, TARGET_IDS, build_graph,
                    edt_radius_all, node_tangents, read_centerline, read_ct, read_mask,
                    root_tree, to_world, vessel_mask, vessel_nodes)


class Report:
    """Накопитель строк отчёта: печатает и сохраняет."""

    def __init__(self, title: str, path):
        self.title = title
        self.path = path
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)
        print(text)

    def save(self) -> None:
        self.path = common._ensure_parent(self.path)
        self.path.write_text(self.title + "\n" + "=" * len(self.title) + "\n\n"
                             + "\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\nотчёт: {self.path}")


def qc_scan(scan_id: str) -> tuple[dict, list[dict]]:
    """Проверки чтения/дерева и радиусы целевых сосудов одного скана."""
    ct, aff = read_ct(scan_id)
    mask, aff_m, zooms = read_mask(scan_id)
    common.check_geometry(aff, aff_m)

    # KD-tree всех вокселей маски: ближайший воксель и его метка.
    vox = np.argwhere(mask > 0)
    vox_label = mask[vox[:, 0], vox[:, 1], vox[:, 2]]
    kd = cKDTree(to_world(aff, vox))

    scan = {"scan_id": scan_id, "voxels": int(len(vox)), "trees": {}}
    per_vessel: list[dict] = []

    lines = {t: read_centerline(scan_id, t) for t in ("left", "right")}
    labels_present = sorted(int(v) for v in np.unique(mask) if v)
    vms = {lab: vessel_mask(mask, aff, zooms, [lab]) for lab in labels_present}
    all_pts = np.vstack([lines[t].points for t in ("left", "right")])
    r_edt_all, snap_all = edt_radius_all(mask, aff, zooms, all_pts)

    for tree in ("left", "right"):
        cl = lines[tree]
        # Ближайший воксель маски: согласие меток и «snap» центрлинии к маске.
        dist, j = kd.query(cl.points)
        agree = float((vox_label[j] == cl.labels).mean() * 100)
        # «Внутри» здесь — не пересечение, а малое расстояние до маски (точка
        # центрлинии непрерывна и может лежать чуть вне дискретного лумена).
        inside = float((dist <= max(zooms)).mean() * 100)

        g = build_graph(cl)
        n_comp = nx.number_connected_components(g)
        rooted = root_tree(cl, g)
        unreachable = cl.n_points - len(rooted.order)

        # Компоненты, не содержащие ни одного устья (start_points): это «оторванные»
        # куски разметки, в дерево не попадают и к целевым сосудам не относятся.
        root_set = set(cl.roots)
        orphan = [c for c in nx.connected_components(g) if not (c & root_set)]

        scan["trees"][tree] = {
            "n_points": cl.n_points, "n_cells": len(cl.cells),
            "n_components": int(n_comp), "unreachable": int(unreachable),
            "n_orphan": int(len(orphan)),
            "orphan_sizes": sorted((len(c) for c in orphan), reverse=True),
            "inside_pct": inside, "agree_pct": agree,
            "snap_p50": float(np.median(dist)), "snap_p95": float(np.percentile(dist, 95)),
            "snap_max": float(dist.max()),
            "snap_out_pct": float((dist > MAX_SNAP_MM).mean() * 100),
        }

        tang = node_tangents(rooted)
        # r_edt_all посчитан одним EDT на весь скан; вырезаем кусок этого дерева.
        start = 0 if tree == "left" else lines["left"].n_points
        r_edt_tree = r_edt_all[start:start + cl.n_points]
        for label in TARGET_IDS:
            idx, arc = vessel_nodes(rooted, label)
            if len(idx) < 2:
                continue
            vm = vms.get(label)
            if vm is None:
                continue
            r_sec = np.array([vm.section_radius(cl.points[i], tang[i])[0] for i in idx])
            r_edt = r_edt_tree[idx]
            lo, hi = RADIUS_SANITY
            per_vessel.append({
                "scan_id": scan_id, "vessel": SEGMENT_NAMES[label], "tree": tree,
                "n_nodes": int(len(idx)), "n_orphan": int(len(orphan)),
                "arc_min": float(arc.min()), "arc_max": float(arc.max()),
                "len_mm": float(arc.max() - arc.min()),
                "sec_p5": float(np.percentile(r_sec, 5)),
                "sec_p50": float(np.median(r_sec)),
                "sec_p95": float(np.percentile(r_sec, 95)),
                "edt_p50": float(np.median(r_edt)),
                "frac_outside_sanity": float(((r_sec < lo) | (r_sec > hi)).mean()),
                # Отношение EDT-радиуса к радиусу по сечению (оба — радиусы):
                # ~1 при согласии, 2 при классической путанице диаметр/радиус.
                "r_edt_over_section": float(np.median(r_edt[r_sec > 0] / r_sec[r_sec > 0]))
                if (r_sec > 0).any() else float("nan"),
            })
    return scan, per_vessel


def save_csv(rows: list[dict], path) -> None:
    path = common._ensure_parent(path)
    cols = ["scan_id", "vessel", "tree", "n_nodes", "n_orphan", "arc_min", "arc_max", "len_mm",
            "sec_p5", "sec_p50", "sec_p95", "edt_p50", "frac_outside_sanity", "r_edt_over_section"]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    print(f"CSV: {path}")


def save_png(rows: list[dict], path) -> None:
    if not rows:
        print("PNG: нечего строить (нет целевых сосудов)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=130)
    colors = {"LAD": "#FF2D2D", "LCx": "#2D7DFF", "RCA": "#22DD88"}

    ax = axes[0]
    for name in ("LAD", "LCx", "RCA"):
        s = np.array([r["sec_p50"] for r in rows if r["vessel"] == name])
        e = np.array([r["edt_p50"] for r in rows if r["vessel"] == name])
        if s.size:
            ax.scatter(e, s, s=16, alpha=0.6, color=colors[name], label=name)
    lim = [0, max(2.5, max((r["sec_p50"] for r in rows), default=1) * 1.1)]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("EDT (радиус), медиана, мм")
    ax.set_ylabel("радиус по сечению, медиана, мм")
    ax.set_title("Радиус: сечение против EDT")
    ax.legend(fontsize=8)

    ax = axes[1]
    lens = [r["len_mm"] for r in rows]
    ax.hist(lens, bins=30, color="#00C2A8")
    ax.set_xlabel("длина ветви (arc), мм")
    ax.set_ylabel("сосудов")
    ax.set_title(f"Длины ветвей: p50 {np.median(lens):.0f} мм, max {max(lens):.0f} мм")

    ax = axes[2]
    for name in ("LAD", "LCx", "RCA"):
        s = np.array([r["sec_p50"] for r in rows if r["vessel"] == name])
        if s.size:
            ax.hist(s, bins=25, alpha=0.5, color=colors[name], label=name)
    ax.set_xlabel("радиус по сечению, медиана, мм")
    ax.set_ylabel("сосудов")
    ax.set_title("Распределение радиуса по сосудам")
    ax.legend(fontsize=8)

    fig.suptitle("QC радиусов и длин целевых сосудов", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = common._ensure_parent(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {path}")


def main() -> int:
    args = common.parse_common_args("QC и анализ данных")
    scan_ids = common.resolve_scan_ids(args)
    rep = Report(f"QC · ImageCAS-X · split {args.split} · сканов {len(scan_ids)}",
                 args.out / "qc" / f"report_{args.split}.txt")
    rep.add(f"сканов: {len(scan_ids)}")

    vessel_rows: list[dict] = []
    orphan_scans: list[str] = []
    for scan_id in tqdm(scan_ids, desc="qc", unit="scan", mininterval=5.0):
        scan, rows = qc_scan(scan_id)
        vessel_rows.extend(rows)
        has_orphan = False
        for tree, t in scan["trees"].items():
            if t["n_orphan"]:
                has_orphan = True
            rep.add(f"{scan_id:>5} {tree:>5}: точек {t['n_points']:>5}, ячеек {t['n_cells']:>3}, "
                    f"компонент {t['n_components']}, недостижимо {t['unreachable']:>3}, "
                    f"оторванных {t['n_orphan']} {t['orphan_sizes']}, "
                    f"внутри {t['inside_pct']:>5.1f}%, метки {t['agree_pct']:>5.1f}%, "
                    f"snap p50/p95/max {t['snap_p50']:.2f}/{t['snap_p95']:.2f}/{t['snap_max']:.2f}, "
                    f"> {MAX_SNAP_MM:g} мм: {t['snap_out_pct']:.1f}%")
        if has_orphan:
            orphan_scans.append(scan_id)

    rep.add("")
    rep.add(f"Сканы с оторванными компонентами (без устья): "
            f"{len(orphan_scans)} — {', '.join(orphan_scans) or 'нет'}")

    rep.add("")
    rep.add("Целевые сосуды (медианы радиуса и длины):")
    for name in ("LAD", "LCx", "RCA"):
        rs = [r for r in vessel_rows if r["vessel"] == name]
        if not rs:
            rep.add(f"  {name}: нет")
            continue
        rep.add(f"  {name}: сосудов {len(rs)}, "
                f"радиус сечение p5/p50/p95 = "
                f"{np.percentile([r['sec_p5'] for r in rs], 5):.2f}/"
                f"{np.median([r['sec_p50'] for r in rs]):.2f}/"
                f"{np.percentile([r['sec_p95'] for r in rs], 95):.2f} мм, "
                f"EDT медиана {np.median([r['edt_p50'] for r in rs]):.2f} мм, "
                f"вне sanity {100 * np.mean([r['frac_outside_sanity'] for r in rs]):.1f}%, "
                f"длина p5/p50/p95 = "
                f"{np.percentile([r['len_mm'] for r in rs], 5):.0f}/"
                f"{np.median([r['len_mm'] for r in rs]):.0f}/"
                f"{np.percentile([r['len_mm'] for r in rs], 95):.0f} мм")
    rep.save()
    save_csv(vessel_rows, args.out / "qc" / f"vessels_{args.split}.csv")
    save_png(vessel_rows, args.out / "qc" / f"{args.split}_qc.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
