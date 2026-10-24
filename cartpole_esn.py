import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Резервуарный контроллер для перевёрнутого маятника на тележке.
# Резервуар видит только положение тележки x и угол θ — без скоростей.
# Скорости он должен «вычислить» сам из памяти о прошлых входах.
# Обучаем только выходной слой: имитируем учителя (LQR-регулятор).

np.random.seed(42)

# ==========================================
# ШАГ 1: ФИЗИКА ТЕЛЕЖКИ С МАЯТНИКОМ
# ==========================================

g = 9.81       # ускорение свободного падения
m_cart = 1.0   # масса тележки, кг
m_pole = 0.1   # масса стержня, кг
l = 0.5        # половина длины стержня, м
M = m_cart + m_pole

dt = 0.02      # шаг управления, 50 Гц
F_max = 10.0   # ограничение силы мотора, Н
x_lim = 2.4    # края рельса, м
th_lim = 0.7   # угол падения, рад (~40°)
th0_range = (0.0, 0.2)  # модуль начального угла, рад


def deriv(s, F):
    x, x_dot, th, th_dot = s
    sn, cs = np.sin(th), np.cos(th)
    temp = (F + m_pole * l * th_dot**2 * sn) / M
    th_acc = (g * sn - cs * temp) / (l * (4 / 3 - m_pole * cs**2 / M))
    x_acc = temp - m_pole * l * th_acc * cs / M
    return np.array([x_dot, x_acc, th_dot, th_acc])


def physics_step(s, F):
    # Рунге-Кутта 4-го порядка
    k1 = deriv(s, F)
    k2 = deriv(s + dt / 2 * k1, F)
    k3 = deriv(s + dt / 2 * k2, F)
    k4 = deriv(s + dt * k3, F)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def failed(s):
    return abs(s[0]) > x_lim or abs(s[2]) > th_lim


def random_state():
    u = np.random.uniform(-1, 1)
    th = np.sign(u) * (th0_range[0] + (th0_range[1] - th0_range[0]) * abs(u))
    return np.array([np.random.uniform(-1, 1), 0.0, th, 0.0])

# ==========================================
# ШАГ 2: УЧИТЕЛЬ — LQR-РЕГУЛЯТОР
# ==========================================

# Линеаризуем систему в верхней точке численным дифференцированием
eps = 1e-6
A = np.zeros((4, 4))
for i in range(4):
    e = np.zeros(4)
    e[i] = eps
    A[:, i] = (deriv(e, 0) - deriv(-e, 0)) / (2 * eps)
B = ((deriv(np.zeros(4), eps) - deriv(np.zeros(4), -eps)) / (2 * eps)).reshape(4, 1)

# Дискретизация: Ad = exp(A·dt), Bd = ∫exp(A·τ)dτ·B
Ad = np.eye(4)
Bd = np.zeros((4, 4))
term = np.eye(4)
for k in range(1, 20):
    Bd += term * dt / k
    term = term @ A * dt / k
    Ad += term
Bd = Bd @ B

# Итерации дискретного уравнения Риккати
Q = np.diag([1.0, 0.1, 10.0, 0.1])
R = np.array([[0.1]])
P = Q.copy()
for _ in range(3000):
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    P = Q + Ad.T @ P @ (Ad - Bd @ K)
K = K.ravel()

print(f"LQR: K = {np.round(K, 2)}")


def teacher(s):
    # Учитель видит полное состояние, включая скорости
    return float(np.clip(-K @ s, -F_max, F_max))

# ==========================================
# ШАГ 3: СОЗДАНИЕ РЕЗЕРВУАРА
# ==========================================

# Вместо случайной сети — нервная система нематоды C. elegans:
# 300 нейронов гермафродита и все синапсы между ними (Cook et al., 2019,
# данные OpenWorm/c302). Мышцы и другие клетки выброшены.
# Химический синапс A→B даёт вес по числу синапсов; ГАМК-ергические
# нейроны тормозят (вес со знаком минус). Щелевые контакты (gap junctions)
# симметричны и всегда возбуждают.

reservoir = "connectome"  # "connectome" или "random" — для сравнения
input_to = "sensory"      # "sensory" — вход только в сенсорные нейроны, "all" — во все

GABA = {"DD01", "DD02", "DD03", "DD04", "DD05", "DD06",
        "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07",
        "VD08", "VD09", "VD10", "VD11", "VD12", "VD13",
        "RMED", "RMEL", "RMER", "RMEV", "RIS", "AVL", "DVB"}
SENSORY = {"ADE", "ADF", "ADL", "AFD", "ALM", "ALN", "AQR", "ASE", "ASG",
           "ASH", "ASI", "ASJ", "ASK", "AVM", "AWA", "AWB", "AWC", "BAG",
           "CEP", "FLP", "IL1", "IL2", "OLL", "OLQ", "PDE", "PHA", "PHB",
           "PHC", "PLM", "PLN", "PQR", "PVD", "PVM", "SDQ", "URB", "URX", "URY"}


def load_connectome(path):
    edges = [line.strip().split(",") for line in open(path).readlines()[1:]]
    names = sorted({e[0] for e in edges} | {e[1] for e in edges})
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for pre, post, count, kind in edges:
        w = np.log1p(int(count))  # 1 синапс и 50 синапсов — не в 50 раз сильнее
        if kind == "chemical":
            W[idx[post], idx[pre]] += -w if pre in GABA else w
        else:
            W[idx[post], idx[pre]] += w / 2  # контакт записан в обе стороны
            W[idx[pre], idx[post]] += w / 2
    return names, W


names, W_res = load_connectome(os.path.join(os.path.dirname(__file__), "celegans_connectome.csv"))
N = len(names)
sparsity = 0.1
spectral_radius = 0.9
lr = 0.6          # leak rate: быстрая память нужна для оценки скоростей
in_scale = 1.0
reg = 1e-3
washout = 50

if reservoir == "random":
    W_res = np.random.uniform(-1.0, 1.0, (N, N))
    W_res *= np.random.rand(N, N) < sparsity
W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W_res)))

W_in = np.random.uniform(-1.0, 1.0, (N, 2)) * in_scale
if input_to == "sensory":
    sensory = np.array([n[:3] in SENSORY for n in names])
    W_in[~sensory] = 0.0
bias = np.random.uniform(-0.2, 0.2, N)

obs_noise = 0.002  # шум датчиков, м и рад
kick_prob = 0.005     # вероятность толчка маятника на шаге
kick_every = None     # (мин, макс) с между толчками; если задано, вместо kick_prob
kick_strength = 0.8   # сила толчка, рад/с
obs_scale = np.array([x_lim, 0.2])

print(f"Резервуар: {reservoir}, нейронов: {N}, связей: {np.count_nonzero(W_res)}")
if input_to == "sensory":
    print(f"Вход только в сенсорные нейроны: {sensory.sum()} из {N}")
print(f"Спектральный радиус: {spectral_radius}, leak rate: {lr}")


def observe(s):
    # Датчики: только x и θ, с шумом
    o = np.array([s[0], s[2]]) + np.random.normal(0, obs_noise, 2)
    return o / obs_scale


def reservoir_step(r, o):
    return (1 - lr) * r + lr * np.tanh(W_res @ r + W_in @ o + bias)


def reset_episode():
    s = random_state()
    r = np.zeros(N)
    for _ in range(washout):
        r = reservoir_step(r, observe(s))
    return s, r

# ==========================================
# ШАГ 4: ПРОГОН СИСТЕМЫ
# ==========================================


def run(steps, W_out=None, beta=1.0, explore=True, kicks=True, record=False):
    """beta — доля учителя в применяемой силе (1 = управляет учитель,
    0 = управляет резервуар). explore — шум в силе для разнообразия данных."""
    states, targets = [], []
    log = {"s": [], "u": [], "u_teacher": [], "r": [], "fail": []}
    s, r = reset_episode()
    ou = 0.0
    fails = 0
    next_kick = np.random.uniform(*kick_every) / dt if kick_every else None

    for _ in range(steps):
        r = reservoir_step(r, observe(s))
        features = np.append(r, 1.0)
        u_teacher = teacher(s)
        states.append(features)
        targets.append(u_teacher)

        if W_out is None:
            u_esn = u_teacher
        else:
            u_esn = float(np.clip(W_out @ features, -F_max, F_max))

        u = beta * u_teacher + (1 - beta) * u_esn
        if explore:
            ou += -0.2 * ou + 0.8 * np.random.normal()  # шум Орнштейна-Уленбека
            u += 1.5 * ou

        if record:
            log["s"].append(s.copy())
            log["u"].append(u)
            log["u_teacher"].append(u_teacher)
            log["r"].append(r.copy())

        s = physics_step(s, u)
        if next_kick is None:
            kick_now = kicks and np.random.rand() < kick_prob
        else:
            next_kick -= 1
            kick_now = kicks and next_kick <= 0
            if next_kick <= 0:
                next_kick = np.random.uniform(*kick_every) / dt
        if kick_now:
            s[3] += np.random.normal() * kick_strength  # случайный толчок маятника

        if failed(s):
            fails += 1
            if record:
                log["fail"].append(len(log["s"]))
            s, r = reset_episode()

    return np.array(states), np.array(targets), fails, log


def ridge(S, Y):
    return np.linalg.solve(S.T @ S + reg * np.eye(S.shape[1]), S.T @ Y)

if __name__ == "__main__":
    # ==========================================
    # ШАГ 5: ОБУЧЕНИЕ ВЫХОДНОГО СЛОЯ
    # ==========================================

    # 5 минут симуляции: управляет учитель, резервуар только наблюдает
    S, Y, fails, _ = run(15000, beta=1.0)
    W_out = ridge(S, Y)
    rmse_train = np.sqrt(np.mean((S @ W_out - Y) ** 2))
    print(f"Обучение (учитель): падений {fails}, RMSE силы {rmse_train:.3f} Н")

    # DAgger: управляет смесь учитель/резервуар, учитель размечает
    # состояния, в которые попадает уже сам резервуар
    S2, Y2, fails, _ = run(10000, W_out=W_out, beta=0.5)
    S = np.vstack([S, S2])
    Y = np.concatenate([Y, Y2])
    W_out = ridge(S, Y)
    print(f"DAgger: падений {fails}, всего {len(Y)} примеров")

    # ==========================================
    # ШАГ 6: ТЕСТ — РЕЗЕРВУАР УПРАВЛЯЕТ САМ
    # ==========================================

    test_steps = 3000  # 60 секунд
    _, _, fails, log = run(test_steps, W_out=W_out, beta=0.0,
                           explore=False, kicks=True, record=True)
    u_esn = np.array(log["u"])
    u_t = np.array(log["u_teacher"])
    print(f"Тест 60 с с толчками: падений {fails}")
    print(f"RMSE резервуар vs учитель: {np.sqrt(np.mean((u_esn - u_t) ** 2)):.3f} Н")

    # Для сравнения: тот же линейный выход, но без резервуара — только [x, θ, 1]
    _, _, _, ref = run(15000, beta=1.0, record=True)
    X_lin = np.column_stack([np.array(ref["s"])[:, [0, 2]] / obs_scale,
                             np.ones(len(ref["s"]))])
    W_lin = ridge(X_lin, np.array(ref["u_teacher"]))

    s = random_state()
    lin_fails = 0
    for _ in range(test_steps):
        o = observe(s)
        u = float(np.clip(W_lin @ np.append(o, 1.0), -F_max, F_max))
        s = physics_step(s, u)
        if failed(s):
            lin_fails += 1
            s = random_state()
    print(f"Без резервуара (только x и угол): падений за 60 с {lin_fails}")

    # ==========================================
    # ШАГ 7: ВИЗУАЛИЗАЦИЯ
    # ==========================================

    traj = np.array(log["s"])
    acts = np.array(log["r"])
    t = np.arange(test_steps) * dt

    fig = plt.figure(figsize=(14, 8))
    ax_cart = fig.add_subplot(2, 2, 1)
    ax_res = fig.add_subplot(2, 2, 2)
    ax_th = fig.add_subplot(2, 2, 3)
    ax_f = fig.add_subplot(2, 2, 4)

    # Тележка и маятник
    ax_cart.set_xlim(-x_lim - 0.3, x_lim + 0.3)
    ax_cart.set_ylim(-0.4, 1.3)
    ax_cart.set_aspect("equal")
    ax_cart.axhline(0, color="gray", linewidth=2)
    for edge in (-x_lim, x_lim):
        ax_cart.axvline(edge, color="red", alpha=0.3)
    cart = plt.Rectangle((0, -0.1), 0.4, 0.2, color="tab:blue")
    ax_cart.add_patch(cart)
    pole, = ax_cart.plot([], [], color="tab:orange", linewidth=5, solid_capstyle="round")
    force_arrow, = ax_cart.plot([], [], color="tab:green", linewidth=3)
    info = ax_cart.text(-x_lim, 1.15, "", fontsize=10)
    ax_cart.set_title("Резервуар управляет тележкой")

    # Активность резервуара: 300 нейронов как картинка 15×20
    img = ax_res.imshow(acts[0].reshape(15, 20), cmap="coolwarm", vmin=-1, vmax=1)
    ax_res.set_title("Нейроны C. elegans" if reservoir == "connectome" else "Состояния нейронов резервуара")
    ax_res.set_xticks([])
    ax_res.set_yticks([])
    fig.colorbar(img, ax=ax_res, fraction=0.035)

    # Угол и положение
    ax_th.plot(t, traj[:, 2], label="θ, рад", color="tab:orange")
    ax_th.plot(t, traj[:, 0], label="x, м", color="tab:blue", alpha=0.7)
    for k in log["fail"]:
        ax_th.axvline(k * dt, color="red", linestyle=":")
    cursor_th = ax_th.axvline(0, color="black", alpha=0.5)
    ax_th.set_xlabel("Время, с")
    ax_th.legend(loc="upper right")
    ax_th.grid(True, alpha=0.3)

    # Сила: резервуар vs что сделал бы учитель
    ax_f.plot(t, u_t, label="Учитель LQR (полное состояние)", color="tab:blue", alpha=0.65)
    ax_f.plot(t, u_esn, label="Резервуар (только x, θ)", color="tab:red", linestyle="--")
    cursor_f = ax_f.axvline(0, color="black", alpha=0.5)
    ax_f.set_xlabel("Время, с")
    ax_f.set_ylabel("Сила, Н")
    ax_f.legend(loc="upper right")
    ax_f.grid(True, alpha=0.3)

    plt.tight_layout()

    frame_step = 2


    def update(i):
        k = i * frame_step
        x, _, th, _ = traj[k]
        cart.set_x(x - 0.2)
        pole.set_data([x, x + 2 * l * np.sin(th)], [0, 2 * l * np.cos(th)])
        force_arrow.set_data([x, x + u_esn[k] * 0.05], [-0.25, -0.25])
        info.set_text(f"t = {t[k]:5.2f} с   θ = {np.degrees(th):+5.1f}°   F = {u_esn[k]:+5.1f} Н")
        img.set_data(acts[k].reshape(15, 20))
        cursor_th.set_xdata([t[k], t[k]])
        cursor_f.set_xdata([t[k], t[k]])
        return cart, pole, force_arrow, info, img, cursor_th, cursor_f


    anim = FuncAnimation(fig, update, frames=test_steps // frame_step,
                         interval=dt * frame_step * 1000, blit=False)
    plt.show()
