import argparse
import json
import os
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
#   python compare_esn.py                  # 50 сидов 2000–2049, гистограмма falls_hist_50.png
#   python compare_esn.py --seeds 100

MODES = {"connectome": ("Коннектом C. elegans", "#2a78d6"),
         "random": ("Случайный резервуар", "#eb6834")}


def collect(mode, seeds, storage, cache):
    # Прогоны долгие (~7 с на сид), поэтому результаты кешируются в json
    if os.path.exists(cache):
        res = json.load(open(cache))
        if [r["seed"] for r in res] == seeds:
            return res
    p = optuna.load_study(study_name=f"esn_{mode}", storage=storage).best_trial.params
    res = []
    for s in seeds:
        fails, rmse = te.evaluate(mode, p, [s])
        res.append({"seed": s, "fails": fails, "rmse": rmse})
    json.dump(res, open(cache, "w"))
    return res


def report(c, r, c_rmse, r_rmse):
    for name, x, e in (("коннектом", c, c_rmse), ("случайный", r, r_rmse)):
        q1, q3 = np.percentile(x, [25, 75])
        print(f"{name}: среднее {x.mean():.2f}, медиана {np.median(x):.1f}, IQR {q1:.0f}–{q3:.0f}, "
              f"σ {x.std(ddof=1):.2f}, дисперсия/среднее {x.var(ddof=1) / x.mean():.2f}, "
              f"0 падений {np.mean(x == 0) * 100:.0f}%, ≥6 {np.mean(x >= 6) * 100:.0f}%, "
              f"макс {x.max():.0f}, RMSE {e.mean():.3f} Н")

    d = c - r
    print(f"\nКоннектом − случайный: среднее {d.mean():+.2f}, медиана {np.median(d):+.1f}; "
          f"лучше в {np.sum(d < 0)}, хуже в {np.sum(d > 0)}, ничьих {np.sum(d == 0)}")
    print(f"Уилкоксон (парный, Pratt): p = {stats.wilcoxon(c, r, zero_method='pratt').pvalue:.3f}")
    print(f"Манн–Уитни (непарный):     p = {stats.mannwhitneyu(c, r).pvalue:.3f}")
    print(f"Уилкоксон по RMSE:         p = {stats.wilcoxon(c_rmse, r_rmse).pvalue:.3f}")
    print(f"Браун–Форсайт (разброс):   p = {stats.levene(c, r, center='median').pvalue:.3f}")

    # Бутстрап по сидам: пары коннектом/случайный сохраняются
    idx = np.random.default_rng(0).integers(0, len(d), (100000, len(d)))
    for lab, b in (("разница средних", c[idx].mean(1) - r[idx].mean(1)),
                   ("разница медиан", np.median(c[idx], 1) - np.median(r[idx], 1)),
                   ("разница σ", c[idx].std(1, ddof=1) - r[idx].std(1, ddof=1))):
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"Бутстрап 95% ДИ, {lab}: [{lo:+.2f}, {hi:+.2f}]")


def histogram(falls, path):
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    top = int(max(x.max() for x in falls.values()))
    bins = np.arange(top + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=surf)
    ax.set_facecolor(surf)
    w = 0.4
    for k, (mode, x) in enumerate(falls.items()):
        name, col = MODES[mode]
        cnt = np.bincount(x.astype(int), minlength=top + 1)
        ax.bar(bins + (k - 0.5) * w, cnt, width=w - 0.04, color=col, edgecolor=surf, linewidth=1,
               label=f"{name}: медиана {np.median(x):.0f}, среднее {x.mean():.2f}", zorder=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(ink2)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    ax.tick_params(colors=ink2, length=0)
    ax.set_xticks(bins)
    n = len(next(iter(falls.values())))
    ax.set_xlabel(f"Падений за {te.TEST_STEPS * te.ce.dt:.0f} с", color=ink2)
    ax.set_ylabel(f"Число прогонов (из {n})", color=ink2)
    ax.set_title(f"Распределение падений лучших конфигураций, {n} новых сидов",
                 color=ink, loc="left", fontsize=12)
    for t in ax.legend(frameon=False, loc="upper right").get_texts():
        t.set_color(ink)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=surf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=2000,
                    help="не пересекаться с подбором (100–102) и финальной оценкой (1000–1009)")
    ap.add_argument("--storage", default="sqlite:///tune_esn.db")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    res = {m: collect(m, seeds, args.storage, f"compare_{m}_{args.seeds}.json") for m in MODES}
    falls = {m: np.array([x["fails"] for x in v]) for m, v in res.items()}
    rmse = {m: np.array([x["rmse"] for x in v]) for m, v in res.items()}

    report(falls["connectome"], falls["random"], rmse["connectome"], rmse["random"])
    histogram(falls, f"falls_hist_{args.seeds}.png")
