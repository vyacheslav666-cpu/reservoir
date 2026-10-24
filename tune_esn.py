import argparse
import os
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # см. cartpole_esn.py; до импорта numpy
import numpy as np
import optuna

import cartpole_esn as ce

# Раздельный подбор гиперпараметров для резервуаров: коннектом C. elegans,
# случайная сеть того же размера и два контроля из самого коннектома —
# shuffled (те же веса на случайных местах) и rewired (рёбра перемешаны,
# степени нейронов сохранены). Бюджет одинаковый, W_in и bias общие —
# режимы отличаются только матрицей W_res.
# Обучение и тест — те же run() и ridge() из cartpole_esn.py.
#
#   python tune_esn.py --trials 200
#   python tune_esn.py --mode random --trials 200    # только один режим
#   python tune_esn.py --mode shuffled --trials 200  # контроль топологии
#   python tune_esn.py --final-only                  # переоценить лучшие из базы

SPACE = {
    "spectral_radius": (0.3, 1.1),
    "lr": (0.2, 1.0),
    "in_scale": (0.1, 1.5),   # логарифмическая шкала
    "reg": (1e-9, 1e-2),      # логарифмическая шкала
    "sparsity": (0.05, 0.2),  # только для случайного резервуара
}
LOG = {"in_scale", "reg"}

# Тяжёлые условия — и при обучении, и в тесте. Подобраны так, чтобы учитель
# (полное состояние без шума) почти не падал, а резервуары падали несколько раз:
# тогда падения различают конфигурации. Старт 30–45° учитель не вытягивает —
# тележка вылетает за рельс, — поэтому 20–35°.
ce.th0_range = (np.radians(20), np.radians(35))
ce.kick_every = (1.0, 2.0)   # толчок каждые 1–2 с вместо раза в ~4 с
ce.kick_strength = 1.0       # было 0.8
ce.obs_noise = 0.02          # было 0.002: 2 см и ~1.1° на датчиках
TEST_STEPS = 3000            # 60 с

# Общие для обоих режимов: базовые входные веса и смещения.
# W_in умножается на in_scale из trial, маска сенсорных нейронов та же.
shared = np.random.default_rng(0)
W_in_base = shared.uniform(-1.0, 1.0, (ce.N, 2))
if ce.input_to == "sensory":
    sensory = np.array([n[:3] in ce.SENSORY for n in ce.names])
    W_in_base[~sensory] = 0.0
bias = shared.uniform(-0.2, 0.2, ce.N)

# Случайный резервуар: одни и те же числа для всех trials, от sparsity зависит
# только порог маски — сеть с sparsity 0.1 содержит все связи сети с 0.05
W_conn = ce.load_connectome(os.path.join(os.path.dirname(ce.__file__), "celegans_connectome.csv"))[1]
rand = np.random.default_rng(1)
W_rand_values = rand.uniform(-1.0, 1.0, (ce.N, ce.N))
W_rand_mask = rand.random((ce.N, ce.N))

# Контроли из коннектома: при подборе одна матрица на все trials (сид 1),
# как у случайной сети; net_seed задаёт другую перестановку
CONTROLS = {"shuffled": ce.shuffle_edges, "rewired": ce.rewire}
_controls = {}


def control_matrix(mode, net_seed):
    if (mode, net_seed) not in _controls:
        _controls[mode, net_seed] = CONTROLS[mode](W_conn, np.random.default_rng(net_seed))
    return _controls[mode, net_seed]


def build(mode, p, net_seed=1):
    if mode == "connectome":
        W = W_conn.copy()
    elif mode in CONTROLS:
        W = control_matrix(mode, net_seed).copy()
    else:
        W = W_rand_values * (W_rand_mask < p["sparsity"])
    W *= p["spectral_radius"] / np.max(np.abs(np.linalg.eigvals(W)))
    ce.W_res = W
    ce.W_in = W_in_base * p["in_scale"]
    ce.bias = bias
    ce.lr = p["lr"]
    ce.reg = p["reg"]


def run_seed(mode, p, seed, net_seed=1, train_steps=15000, dagger_steps=10000,
             test_steps=TEST_STEPS):
    """Учитель → DAgger → тест с толчками, как в cartpole_esn.py.
    Возвращает число падений и RMSE силы резервуар vs учитель."""
    build(mode, p, net_seed)
    np.random.seed(seed)
    S, Y, _, _ = ce.run(train_steps, beta=1.0)
    W_out = ce.ridge(S, Y)
    S2, Y2, _, _ = ce.run(dagger_steps, W_out=W_out, beta=0.5)
    W_out = ce.ridge(np.vstack([S, S2]), np.concatenate([Y, Y2]))
    _, _, f, log = ce.run(test_steps, W_out=W_out, beta=0.0,
                          explore=False, kicks=True, record=True)
    return f, float(np.sqrt(np.mean((np.array(log["u"]) - np.array(log["u_teacher"])) ** 2)))


JOBS = os.cpu_count()  # процессов для прогона сидов; 1 — всё в текущем процессе
_pool = None


def run_seeds(mode, p, seeds, net_seeds=None, **steps):
    """[(падения, RMSE)] по сидам. Сиды идут параллельно в JOBS процессах:
    каждый прогон начинается с np.random.seed(seed), так что результат не
    зависит от порядка и числа процессов. net_seeds — своя матрица контроля
    на каждый сид, по умолчанию 1."""
    global _pool
    net_seeds = net_seeds or [1] * len(seeds)
    if JOBS == 1 or len(seeds) == 1:
        return [run_seed(mode, p, s, n, **steps) for s, n in zip(seeds, net_seeds)]
    if _pool is None:
        _pool = ProcessPoolExecutor(JOBS)  # процессы запускаются по мере надобности
    jobs = [_pool.submit(run_seed, mode, p, s, n, **steps) for s, n in zip(seeds, net_seeds)]
    return [j.result() for j in jobs]


def evaluate(mode, p, seeds, net_seed=1, **steps):
    """Среднее число падений и RMSE по сидам."""
    fails, rmses = zip(*run_seeds(mode, p, seeds, [net_seed] * len(seeds), **steps))
    return float(np.mean(fails)), float(np.mean(rmses))


def score(fails, rmse):
    # Главное — не ронять маятник; RMSE (меньше F_max) различает равные по падениям
    return fails + rmse / ce.F_max


def suggest(trial, mode):
    p = {k: trial.suggest_float(k, *SPACE[k], log=(k in LOG))
         for k in ("spectral_radius", "lr", "in_scale", "reg")}
    if mode == "random":
        p["sparsity"] = trial.suggest_float("sparsity", *SPACE["sparsity"])
    return p


def tune(mode, trials, seeds, storage, sampler_seed):
    study = optuna.create_study(study_name=f"esn_{mode}", storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed))

    def objective(trial):
        fails, rmse = evaluate(mode, suggest(trial, mode), seeds)
        trial.set_user_attr("fails", fails)
        trial.set_user_attr("rmse", rmse)
        return score(fails, rmse)

    left = trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    return study


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["connectome", "random", "shuffled", "rewired", "both"],
                    default="both", help="both — коннектом и случайная сеть")
    ap.add_argument("--trials", type=int, default=200, help="бюджет на каждый режим")
    ap.add_argument("--seeds", type=int, default=3, help="прогонов на trial")
    ap.add_argument("--final-seeds", type=int, default=10, help="прогонов для финальной оценки")
    ap.add_argument("--storage", default="sqlite:///tune_esn.db")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=JOBS, help="процессов для прогона сидов")
    args = ap.parse_args()

    JOBS = args.jobs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    modes = ["connectome", "random"] if args.mode == "both" else [args.mode]
    tune_seeds = list(range(100, 100 + args.seeds))
    # Финальная оценка на других сидах: лучший trial отобран по tune_seeds,
    # на них его результат завышен
    final_seeds = list(range(1000, 1000 + args.final_seeds))

    results = {}
    for mode in modes:
        if args.final_only:
            study = optuna.load_study(study_name=f"esn_{mode}", storage=args.storage)
        else:
            print(f"\n=== {mode}: {args.trials} trials ===")
            study = tune(mode, args.trials, tune_seeds, args.storage, sampler_seed=0)
        best = study.best_trial
        fails, rmse = evaluate(mode, best.params, final_seeds)
        results[mode] = (best, fails, rmse)

    print(f"\nФинальная оценка: {len(final_seeds)} прогонов по {TEST_STEPS * ce.dt:.0f} с")
    for mode, (best, fails, rmse) in results.items():
        params = ", ".join(f"{k}={v:.3g}" for k, v in best.params.items())
        print(f"{mode:>10}: падений {fails:.2f}, RMSE {rmse:.3f} Н "
              f"(при подборе: {best.user_attrs['fails']:.2f}, {best.user_attrs['rmse']:.3f})")
        print(f"{'':>10}  {params}")
