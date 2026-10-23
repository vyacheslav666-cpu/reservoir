import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ШАГ 1: ГЕНЕРАЦИЯ АНАЛОГОВОГО СИГНАЛА
# ==========================================

np.random.seed(42)

t = np.linspace(0, 150, 3000)

pure_signal = np.sin(t) + 0.5 * np.cos(3 * t)
noise = np.random.normal(0, 0.1, len(t))
signal = (pure_signal + noise).reshape(-1, 1)

# Первые 800 точек доступны при обучении.
# Остальные 2200 точек сеть должна предсказать сама.
train_len = 800

X_train = signal[:train_len - 1]
Y_train = signal[1:train_len]

Y_test = signal[train_len:]

# ==========================================
# ШАГ 2: СОЗДАНИЕ РЕЗЕРВУАРА
# ==========================================

N = 100
sparsity = 0.1
spectral_radius = 0.95
lr = 0.3

# Входные веса
W_in = np.random.uniform(-0.1, 0.1, (N, 1))

# Внутренние связи резервуара
W_res = np.random.uniform(-0.5, 0.5, (N, N))

# Оставляем только 10% связей
mask = np.random.rand(N, N) < sparsity
W_res *= mask

# Нормализация спектрального радиуса
eigenvalues = np.linalg.eigvals(W_res)
max_eigenvalue = np.max(np.abs(eigenvalues))

if max_eigenvalue > 0:
    W_res *= spectral_radius / max_eigenvalue

print(f"Нейронов: {N}")
print(f"Доля связей: {sparsity * 100:.0f}%")
print(f"Спектральный радиус: {spectral_radius}")

# ==========================================
# ШАГ 3: ОБУЧЕНИЕ РЕЗЕРВУАРА
# ==========================================

states_matrix = np.zeros((len(X_train), N))
x = np.zeros((N, 1))

# Прогоняем обучающий сигнал через резервуар
for t_step in range(len(X_train)):
    u = X_train[t_step, 0]

    x_target = np.tanh(W_in * u + W_res @ x)
    x = (1 - lr) * x + lr * x_target

    states_matrix[t_step, :] = x.ravel()

# Обучаем только выходной слой методом Ridge-регрессии
# Внутренние веса резервуара остаются неизменными.
reg = 1e-6

A = states_matrix.T @ states_matrix + reg * np.eye(N)
B = states_matrix.T @ Y_train

W_out = np.linalg.solve(A, B).T

print("Обучение завершено!")

# ==========================================
# ШАГ 4: ПОДГОТОВКА К ГЕНЕРАЦИИ
# ==========================================

# После обучения x соответствует обработке signal[798].
# Теперь подаём последнюю известную точку signal[799].
u = signal[train_len - 1, 0]

x_target = np.tanh(W_in * u + W_res @ x)
x = (1 - lr) * x + lr * x_target

# ==========================================
# ШАГ 5: АВТОНОМНЫЙ ПРОГНОЗ
# ==========================================

predictions = []

for t_step in range(len(Y_test)):

    # Читаем прогноз из текущего состояния резервуара
    y_pred = (W_out @ x).item()
    predictions.append(y_pred)

    # Прогноз становится входом следующего шага
    u = y_pred

    # Обновляем состояние резервуара
    x_target = np.tanh(W_in * u + W_res @ x)
    x = (1 - lr) * x + lr * x_target

predictions = np.array(predictions)

print(f"Сгенерировано точек: {len(predictions)}")

# ==========================================
# ШАГ 6: ВИЗУАЛИЗАЦИЯ
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    np.arange(len(Y_test)),
    Y_test.ravel(),
    label="Реальный сигнал (будущее)",
    color="blue",
    alpha=0.65,
    linewidth=1.5
)

plt.plot(
    np.arange(len(predictions)),
    predictions,
    label="Автономный прогноз резервуара",
    color="red",
    linestyle="--",
    linewidth=1.5
)

plt.title(
    f"Echo State Network: {N} нейронов, "
    f"спектральный радиус {spectral_radius}"
)
plt.xlabel("Шаги после окончания обучения")
plt.ylabel("Амплитуда")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# ШАГ 7: ОЦЕНКА РЕЗУЛЬТАТА
# ==========================================

rmse = np.sqrt(np.mean((Y_test.ravel() - predictions) ** 2))

print(f"RMSE на будущем сигнале: {rmse:.4f}")