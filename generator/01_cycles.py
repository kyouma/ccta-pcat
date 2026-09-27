"""01 — измерение циклов в центрилиниях ImageCAS-X.

Для каждого дерева каждого скана строим граф точек (узлы = точки, рёбра =
соседние пары внутри ячеек LINES) и считаем:
  * компоненты связности;
  * параллельные рёбра (одна пара узлов из разных ячеек);
  * циклический ранг E - V + C (число независимых циклов);
  * длины базисных циклов (nx.cycle_basis) в мм;
  * длины параллельных 2-циклов;
  * общие рёбра (входят более чем в одну ячейку).

Порогов не применяем — только полная картина. Сохраняем CSV и гистограммы.

Запуск:
    python 01_cycles.py --split test --limit 20
"""

from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import common
from common import _cell_edges, build_multigraph, read_centerline


def measure_cycles(cl: common.Centerline) -> dict:
    """Топология одного дерева: компоненты, ранг, длины циклов, общие рёбра."""
    multi, simple = build_multigraph(cl)

    n_components = nx.number_connected_components(simple)
    e_multi = multi.number_of_edges()
    e_simple = simple.number_of_edges()
    # Каждое лишнее параллельное ребро (сверх одного) — отдельный 2-цикл.
    n_parallel = e_multi - e_simple
    # Циклический ранг (первое число Бетти) E - V + C не зависит от выбора базиса.
    n_cycles = e_multi - simple.number_of_nodes() + n_components

    # Базисные циклы простого графа (параллельные рёбра схлопнуты). Их ровно
    # n_cycles - n_parallel; длины зависят от выбора базиса, но дают масштаб петель.
    basis_lengths = []
    for cyc in nx.cycle_basis(simple):
        length = 0.0
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            length += simple[a][b]["weight"]
        basis_lengths.append(length)

    # Счётчик рёбер по парам узлов и веса: общие рёбра (в >1 ячейке) и их веса.
    pair_count: Counter = Counter()
    pair_weights: dict = defaultdict(list)
    for a, b, w in _cell_edges(cl):
        key = (a, b) if a < b else (b, a)
        pair_count[key] += 1
        pair_weights[key].append(w)

    # Параллельные рёбра между одной парой узлов: цикл длины w_min + w_extra.
    parallel_lengths = []
    for key, cnt in pair_count.items():
        if cnt > 1:
            ws = sorted(pair_weights[key])
            parallel_lengths.extend(ws[0] + w for w in ws[1:])

    shared = {k: c for k, c in pair_count.items() if c > 1}

    return {
        "scan_id": cl.scan_id,
        "tree": cl.tree,
        "n_points": cl.n_points,
        "n_cells": len(cl.cells),
        "n_components": int(n_components),
        "n_edges": int(e_multi),
        "n_parallel": int(n_parallel),
        "n_cycles": int(n_cycles),
        "basis_lengths": basis_lengths,
        "parallel_lengths": parallel_lengths,
        "n_shared_edges": len(shared),
        "shared_lengths": [float(simple[a][b]["weight"]) for a, b in shared],
    }


def run(scan_ids: list[str]) -> list[dict]:
    rows = []
    for scan_id in tqdm(scan_ids, desc="cycles", unit="scan", mininterval=5.0):
        for tree in ("left", "right"):
            cl = read_centerline(scan_id, tree)
            rows.append(measure_cycles(cl))
    return rows


def save_csv(rows: list[dict], path) -> None:
    path = common._ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("scan_id,tree,n_points,n_cells,n_components,n_edges,n_parallel,"
                "n_cycles,n_shared_edges,cycle_lengths_mm\n")
        for r in rows:
            lengths = r["basis_lengths"] + r["parallel_lengths"]
            f.write(f"{r['scan_id']},{r['tree']},{r['n_points']},{r['n_cells']},"
                    f"{r['n_components']},{r['n_edges']},{r['n_parallel']},"
                    f"{r['n_cycles']},{r['n_shared_edges']},"
                    f"{';'.join(f'{v:.3f}' for v in lengths)}\n")
    print(f"CSV: {path}")


def save_histograms(rows: list[dict], path) -> None:
    n_cycles = np.array([r["n_cycles"] for r in rows])
    all_lengths = np.array([v for r in rows for v in r["basis_lengths"] + r["parallel_lengths"]])
    n_parallel = np.array([r["n_parallel"] for r in rows])
    n_shared = np.array([r["n_shared_edges"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)

    vals, counts = np.unique(n_cycles, return_counts=True)
    axes[0].bar(vals, counts, color="#2D7DFF")
    axes[0].set_title(f"Циклов на дерево (n={len(rows)})\n"
                      f"среднее {n_cycles.mean():.2f}, медиана {np.median(n_cycles):.0f}, "
                      f"макс {n_cycles.max()}")
    axes[0].set_xlabel("число циклов")
    axes[0].set_ylabel("деревьев")

    if all_lengths.size:
        axes[1].hist(all_lengths, bins=40, color="#00C2A8")
        axes[1].axvline(1.0, color="red", ls="--", lw=1, label="1 мм")
        axes[1].legend(fontsize=8)
    axes[1].set_title(f"Длины циклов (n={all_lengths.size})\n"
                      f"p50 {np.median(all_lengths):.2f} мм, "
                      f"p95 {np.percentile(all_lengths, 95):.2f} мм, "
                      f"макс {all_lengths.max():.2f} мм" if all_lengths.size else "Циклов нет")
    axes[1].set_xlabel("длина цикла, мм")
    axes[1].set_ylabel("циклов")

    axes[2].scatter(n_shared, n_parallel, s=14, alpha=0.6, color="#FF2D2D")
    axes[2].set_title("Параллельные рёбра vs общие рёбра")
    axes[2].set_xlabel("общих рёбер (в >1 ячейке)")
    axes[2].set_ylabel("параллельных рёбер")

    fig.suptitle("Циклы в центрилиниях ImageCAS-X", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = common._ensure_parent(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {path}")


def summarize(rows: list[dict]) -> None:
    n_cycles = np.array([r["n_cycles"] for r in rows])
    all_lengths = np.array([v for r in rows for v in r["basis_lengths"] + r["parallel_lengths"]])
    has = n_cycles > 0
    print("\n" + "=" * 78)
    print(f"Итог: деревьев {len(rows)}, с циклами {has.sum()} ({100 * has.mean():.1f}%)")
    print(f"циклов на дерево: mean {n_cycles.mean():.2f}, median {np.median(n_cycles):.0f}, "
          f"p90 {np.percentile(n_cycles, 90):.0f}, max {n_cycles.max()}")
    if all_lengths.size:
        print(f"длины циклов, мм: p5 {np.percentile(all_lengths, 5):.2f}, "
              f"p50 {np.median(all_lengths):.2f}, p95 {np.percentile(all_lengths, 95):.2f}, "
              f"max {all_lengths.max():.2f}")
        print(f"коротких (<1 мм): {int((all_lengths < 1).sum())} из {all_lengths.size}")
    else:
        print("циклов не найдено")
    print("=" * 78)


def main() -> int:
    args = common.parse_common_args("Измерение циклов в центрилиниях")
    scan_ids = common.resolve_scan_ids(args)
    print(f"сканов: {len(scan_ids)} (split {args.split})")

    rows = run(scan_ids)
    save_csv(rows, args.out / "cycles" / f"{args.split}_cycles.csv")
    save_histograms(rows, args.out / "cycles" / f"{args.split}_cycles_hist.png")
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
