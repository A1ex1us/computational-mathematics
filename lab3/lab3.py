import numpy as np
import matplotlib.pyplot as plt


def rk4(x0, t_n, f, h):
    x0 = np.asarray(x0, dtype=np.float64)  # Явно указываем тип float64
    n_steps = int(round(t_n / h))
    trajectory = np.zeros((n_steps + 1, 2), dtype=np.float64)
    trajectory[0] = x0
    t = 0.0

    for i in range(n_steps):
        k1 = h * f(t, trajectory[i])
        k2 = h * f(t + h / 2, trajectory[i] + k1 / 2)
        k3 = h * f(t + h / 2, trajectory[i] + k2 / 2)
        k4 = h * f(t + h, trajectory[i] + k3)

        trajectory[i + 1] = trajectory[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return trajectory


# Параметры системы
alpha = 5.0  # Все параметры как float
beta = 0.002
delta = 0.0006
gamma = 0.5

def lotka_volterra(t, x):
    return np.array([
        alpha * x[0] - beta * x[0] * x[1],
        delta * x[0] * x[1] - gamma * x[1]
    ], dtype=np.float64)


# Генерация начальных условий
initial_conditions = []
for i in range(1, 11):
    for j in range(1, 11):
        x0 = 200 * i
        y0 = 200 * j
        initial_conditions.append([x0, y0])

        # Параметры моделирования
t_n = 100  # Конечное время (настройте при необходимости)
h = 0.05  # Шаг интегрирования

# Вычисление траекторий для всех начальных условий
trajectories = []
for x0 in initial_conditions:
    traj = rk4(np.array(x0), t_n, lotka_volterra, h)
    trajectories.append(traj)

plt.figure(figsize=(10, 6))
for traj in trajectories:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)

stat_points = [(0, 0), (2500/3, 2500)]
colors = ['red', 'green']
labels = ['(0, 0)', '(2500/3, 2500)']

for point, color, label in zip(stat_points, colors, labels):
    plt.scatter(*point, c=color, s=100, label=label, zorder=5)
    plt.text(point[0], point[1], label, fontsize=12, ha='right')

plt.xlabel('Жертвы (x)')
plt.ylabel('Хищники (y)')
plt.title('Фазовые траектории системы Лотки–Вольтерры')
plt.grid(True)
plt.show()


representative_traj = trajectories[0]
time = np.arange(0, t_n + h, h)

plt.figure(figsize=(12, 6))

# График x(t) - жертвы
plt.subplot(1, 2, 1)
plt.plot(time, representative_traj[:, 0], color='orange', label='Жертвы (x)')
plt.xlabel('Время (t)')
plt.ylabel('Численность')
plt.title('Динамика жертв')
plt.grid(True)
plt.legend()

# График y(t) - хищники
plt.subplot(1, 2, 2)
plt.plot(time, representative_traj[:, 1], color='red', label='Хищники (y)')
plt.xlabel('Время (t)')
plt.ylabel('Численность')
plt.title('Динамика хищников')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Метод Ньютона с явным указанием типов
def newton(x_0, f, J, max_iter=100, eps=1e-8):
    x_k = x_0.astype(np.float64)
    residuals = []
    iterations = 0

    for _ in range(max_iter):
        f_x = f(x_k)
        J_x = J(x_k)
        delta_x = np.linalg.solve(J_x, -f_x)

        residuals.append(np.linalg.norm(f_x))
        x_k += delta_x

        if np.linalg.norm(delta_x, ord=np.inf) < eps:
            break
        iterations += 1

    return x_k, iterations, residuals


# Градиентный спуск
def gradient_descent(x_0, f, J, max_iter=1000, eps=1e-8):
    x_k = x_0.astype(np.float64)
    residuals = []
    iterations = 0

    for _ in range(max_iter):
        f_x = f(x_k)
        J_x = J(x_k)
        grad = J_x.T @ f_x

        alpha = 0.01
        x_new = x_k - alpha * grad

        residuals.append(np.linalg.norm(f_x))
        delta_x = x_new - x_k

        if np.linalg.norm(delta_x, ord=np.inf) < eps:
            break

        x_k = x_new
        iterations += 1

    return x_k, iterations, residuals


# Функции для системы Лотки-Вольтерры
def f_lotka_volterra(x):
    return np.array([
        alpha * x[0] - beta * x[0] * x[1],
        delta * x[0] * x[1] - gamma * x[1]
    ], dtype=np.float64)


def J_lotka_volterra(x):
    return np.array([
        [alpha - beta * x[1], -beta * x[0]],
        [delta * x[1], delta * x[0] - gamma]
    ], dtype=np.float64)


# Теоретическая стационарная точка
x_star = gamma / delta
y_star = alpha / beta

# Инициализация матриц
newton_iters = np.zeros((201, 201), dtype=np.float64)
gradient_iters = np.zeros((201, 201), dtype=np.float64)
sup_norm_matrix = np.zeros((201, 201), dtype=np.float64)

# Основной цикл вычислений
for i in range(201):
    for j in range(201):
        x0 = 15.0 * i
        y0 = 15.0 * j

        # Метод Ньютона
        root_newton, iters_newton, _ = newton(
            np.array([x0, y0], dtype=np.float64),
            f_lotka_volterra,
            J_lotka_volterra
        )
        newton_iters[i, j] = iters_newton

        # Градиентный спуск
        root_grad, iters_grad, _ = gradient_descent(
            np.array([x0, y0], dtype=np.float64),
            f_lotka_volterra,
            J_lotka_volterra
        )
        gradient_iters[i, j] = iters_grad

        # Супремум-норма
        dx = abs(root_newton[0] - x_star)
        dy = abs(root_newton[1] - y_star)
        sup_norm_matrix[i, j] = max(dx, dy)

# Визуализация линий уровня
X0_grid, Y0_grid = np.meshgrid(
    np.arange(0, 3001, 15, dtype=np.float64),
    np.arange(0, 3001, 15, dtype=np.float64)
)
plt.contourf(X0_grid, Y0_grid, sup_norm_matrix, levels=50, cmap='viridis')
plt.colorbar(label='Супремум-норма')
plt.scatter(x_star, y_star, c='red', s=50)
plt.title('Линии уровня супремум-нормы')
plt.xlabel('x0')
plt.ylabel('y0')
plt.show()

# Log-log график для репрезентативной точки
x0_repr = np.array([1000.0, 1000.0], dtype=np.float64)
root_newton, _, res_newton = newton(x0_repr, f_lotka_volterra, J_lotka_volterra)
root_grad, _, res_grad = gradient_descent(x0_repr, f_lotka_volterra, J_lotka_volterra)

plt.loglog(res_newton, 'o-', label='Ньютон')
plt.loglog(res_grad, 's-', label='Градиентный спуск')
plt.xlabel('Итерации')
plt.ylabel('Норма невязки')
plt.legend()
plt.grid(True)
plt.show()

# Статистики
print(f"Ньютон: μ = {np.mean(newton_iters):.1f} ± {np.std(newton_iters):.1f}")
print(f"Градиент: μ = {np.mean(gradient_iters):.1f} ± {np.std(gradient_iters):.1f}")