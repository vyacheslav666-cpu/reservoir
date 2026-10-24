import argparse
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # см. cartpole_esn.py; до импорта numpy
import numpy as np
import optuna
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tune_esn as te

# Сравнение лучших конфигураций из tune_esn.db на новых сидах.
# Падения — счётные данные с выбросами, поэтому вместо t-теста:
# Уилкоксон (парный — сиды общие), Манн–Уитни и бутстрап по сидам.
#
#   python compare_esn.py                  # 50 сидов 2000–2049, гистограммы falls_*_50.png
#   python compare_esn.py --seeds 100
#
# Каждый режим сравнивается с коннектомом. Кроме лучших trials, для контролей
# (shuffled, rewired) есть вариант «plug-in»: параметры, подобранные для
# коннектома, и на каждом сиде новая перемешанная матрица. Подбор подогнан
# под коннектом, так что в этом варианте преимущество на его стороне.

MODES = {"connectome": ("Коннектом C. elegans", "#2a78d6"),
         "random": ("Случайный резервуар", "#eb6834"),
         "shuffled": ("Перемешанный коннектом", "#1baf7a"),
         "rewired": ("Перемешанный, степени сохранены", "#eda100")}
PLUGIN = ("shuffled", "rewired")


def collect(mode, seeds, storage, cache, plugin=False):
    # Прогоны кешируются в json
    if os.path.exists(cache):
        res = json.load(open(cache))
        if [r["seed"] for r in res] == seeds:
            return res
    study = "esn_connectome" if plugin else f"esn_{mode}"
    p = optuna.load_study(study_name=study, storage=storage).best_trial.params
    runs = te.run_seeds(mode, p, seeds, net_seeds=seeds if plugin else None)
    res = [{"seed": s, "fails": f, "rmse": r} for s, (f, r) in zip(seeds, runs)]
    json.dump(res, open(cache, "w"))
    return res


def describe(name, x, e):
    q1, q3 = np.percentile(x, [25, 75])
    print(f"{name}\n  среднее {x.mean():.2f}, медиана {np.median(x):.1f}, IQR {q1:.0f}–{q3:.0f}, "
          f"σ {x.std(ddof=1):.2f}, 0 падений {np.mean(x == 0) * 100:.0f}%, "
          f"≥6 {np.mean(x >= 6) * 100:.0f}%, макс {x.max():.0f}, RMSE {e.mean():.3f} Н")


def compare(c, r, c_rmse, r_rmse):
    # c — коннектом, r — режим, с которым он сравнивается
    d = c - r
    print(f"  коннектом − режим: среднее {d.mean():+.2f}, медиана {np.median(d):+.1f}; "
          f"лучше в {np.sum(d < 0)}, хуже в {np.sum(d > 0)}, ничьих {np.sum(d == 0)}")
    print(f"  Уилкоксон (парный, Pratt): p = {stats.wilcoxon(c, r, zero_method='pratt').pvalue:.3f}")
    print(f"  Манн–Уитни (непарный):     p = {stats.mannwhitneyu(c, r).pvalue:.3f}")
    print(f"  Уилкоксон по RMSE:         p = {stats.wilcoxon(c_rmse, r_rmse).pvalue:.3f}")
    print(f"  Браун–Форсайт (разброс):   p = {stats.levene(c, r, center='median').pvalue:.3f}")

    # Бутстрап по сидам: пары сохраняются
    idx = np.random.default_rng(0).integers(0, len(d), (100000, len(d)))
    for lab, b in (("разница средних", c[idx].mean(1) - r[idx].mean(1)),
                   ("разница медиан", np.median(c[idx], 1) - np.median(r[idx], 1))):
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  Бутстрап 95% ДИ, {lab}: [{lo:+.2f}, {hi:+.2f}]")


SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK2, length=0)


def histogram(falls, path):
    top = int(max(x.max() for x in falls.values()))
    bins = np.arange(top + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURF)
    style(ax)
    w = 0.4
    for k, (mode, x) in enumerate(falls.items()):
        name, col = MODES[mode]
        cnt = np.bincount(x.astype(int), minlength=top + 1)
        ax.bar(bins + (k - 0.5) * w, cnt, width=w - 0.04, color=col, edgecolor=SURF, linewidth=1,
               label=f"{name}: медиана {np.median(x):.0f}, среднее {x.mean():.2f}", zorder=3)
    ax.set_xticks(bins)
    n = len(next(iter(falls.values())))
    ax.set_xlabel(f"Падений за {te.TEST_STEPS * te.ce.dt:.0f} с", color=INK2)
    ax.set_ylabel(f"Число прогонов (из {n})", color=INK2)
    ax.set_title(f"Распределение падений лучших конфигураций, {n} новых сидов",
                 color=INK, loc="left", fontsize=12)
    for t in ax.legend(frameon=False, loc="upper right").get_texts():
        t.set_color(INK)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=SURF)


def panels(rows, path):
    # Малые множители: каждый режим в своей панели, коннектом — контур
    # для сравнения в каждой
    top = int(max(x.max() for _, _, x in rows))
    bins = np.arange(top + 1)
    conn = rows[0][2]
    c_cnt = np.bincount(conn.astype(int), minlength=top + 1)
    ymax = max(np.bincount(x.astype(int)).max() for _, _, x in rows) + 1
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 1.9 * len(rows) + 0.6),
                             sharex=True, facecolor=SURF)
    for ax, (label, mode, x) in zip(axes, rows):
        style(ax)
        cnt = np.bincount(x.astype(int), minlength=top + 1)
        ax.bar(bins, cnt, width=0.8, color=MODES[mode][1], edgecolor=SURF, linewidth=1, zorder=3)
        if mode != "connectome":
            ax.step(bins, c_cnt, where="mid", color=INK2, linewidth=1.2, zorder=4,
                    label="коннектом")
            for t in ax.legend(frameon=False, loc="upper right").get_texts():
                t.set_color(INK2)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{label}: медиана {np.median(x):.0f}, среднее {x.mean():.2f}",
                     color=INK, loc="left", fontsize=10)
    axes[-1].set_xticks(bins)
    axes[-1].set_xlabel(f"Падений за {te.TEST_STEPS * te.ce.dt:.0f} с, {len(conn)} новых сидов",
                        color=INK2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=SURF)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=2000,
                    help="не пересекаться с подбором (100–102) и финальной оценкой (1000–1009)")
    ap.add_argument("--storage", default="sqlite:///tune_esn.db")
    ap.add_argument("--jobs", type=int, default=te.JOBS, help="процессов для прогона сидов")
    args = ap.parse_args()

    te.JOBS = args.jobs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    n = args.seeds

    # (подпись, режим, plugin)
    configs = [(MODES[m][0], m, False) for m in MODES]
    configs += [(f"{MODES[m][0]}, параметры коннектома", m, True) for m in PLUGIN]
    res = {}
    for label, mode, plugin in configs:
        cache = f"compare_{mode}_plugin_{n}.json" if plugin else f"compare_{mode}_{n}.json"
        res[label] = (mode, collect(mode, seeds, args.storage, cache, plugin))
    falls = {k: np.array([x["fails"] for x in v]) for k, (_, v) in res.items()}
    rmse = {k: np.array([x["rmse"] for x in v]) for k, (_, v) in res.items()}

    for label in res:
        describe(label, falls[label], rmse[label])
    c_label = MODES["connectome"][0]
    for label in list(res)[1:]:
        print(f"\nКоннектом против: {label}")
        compare(falls[c_label], falls[label], rmse[c_label], rmse[label])

    histogram({m: falls[MODES[m][0]] for m in ("connectome", "random")}, f"falls_hist_{n}.png")
    panels([(label, mode, falls[label]) for label, (mode, _) in res.items()], f"falls_controls_{n}.png")
