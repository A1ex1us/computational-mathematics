import numpy as np
import matplotlib.pyplot as plt

# Функции численного дифференцирования
def diff2(x_0, h, f):
    return (f(x_0 + h) - f(x_0 - h)) / (2 * h)

def diff4(x_0, h, f):
    return (1/(12*h)) * (-f(x_0 - 2*h) + 8*f(x_0 - h) - 8*f(x_0 + h) + f(x_0 + 2*h))

# Определение функций
def g1(x):
    return x * np.exp(x)

def g3(x):
    return np.sin(np.pi / x)

# Аналитические производные
def g1_prime(x):
    return np.exp(x) * (1 + x)

def g3_prime(x):
    return - (np.pi / x**2) * np.cos(np.pi / x)

# Параметры
x0_g1 = 3.0
x0_g3 = 0.01
h_values = np.logspace(-15, 0, 100)
exact_g1 = g1_prime(x0_g1)  # 4 * e^3 ≈ 80.342
exact_g3 = g3_prime(x0_g3)  # -10000 * π ≈ -31415.9265

# Вычисление погрешностей
errors_g1_diff2 = []
errors_g1_diff4 = []
errors_g3_diff2 = []
errors_g3_diff4 = []

for h in h_values:
    num_g1_diff2 = diff2(x0_g1, h, g1)
    num_g1_diff4 = diff4(x0_g1, h, g1)
    num_g3_diff2 = diff2(x0_g3, h, g3)
    num_g3_diff4 = diff4(x0_g3, h, g3)
    errors_g1_diff2.append(np.abs(num_g1_diff2 - exact_g1))
    errors_g1_diff4.append(np.abs(num_g1_diff4 - exact_g1))
    errors_g3_diff2.append(np.abs(num_g3_diff2 - exact_g3))
    errors_g3_diff4.append(np.abs(num_g3_diff4 - exact_g3))

# Построение графиков
plt.figure(figsize=(10, 6))

# График для g1
plt.subplot(1, 2, 1)
plt.loglog(h_values, errors_g1_diff2, 'r--', label='diff2')
plt.loglog(h_values, errors_g1_diff4, 'b-', label='diff4')
min_idx_g1_diff2 = np.argmin(errors_g1_diff2)
min_idx_g1_diff4 = np.argmin(errors_g1_diff4)
h_opt_g1_diff2 = h_values[min_idx_g1_diff2]
h_opt_g1_diff4 = h_values[min_idx_g1_diff4]
plt.axvline(x=h_opt_g1_diff2, color='r', linestyle=':', label=f'h_opt diff2 ≈ {h_opt_g1_diff2:.1e}')
plt.axvline(x=h_opt_g1_diff4, color='b', linestyle=':', label=f'h_opt diff4 ≈ {h_opt_g1_diff4:.1e}')
plt.xlabel('Шаг дифференцирования h')
plt.ylabel('Абсолютная погрешность')
plt.title('Погрешность для g1(x) при x=3')
plt.grid(True)
plt.legend()

# График для g3
plt.subplot(1, 2, 2)
plt.loglog(h_values, errors_g3_diff2, 'r--', label='diff2')
plt.loglog(h_values, errors_g3_diff4, 'b-', label='diff4')
min_idx_g3_diff2 = np.argmin(errors_g3_diff2)
min_idx_g3_diff4 = np.argmin(errors_g3_diff4)
h_opt_g3_diff2 = h_values[min_idx_g3_diff2]
h_opt_g3_diff4 = h_values[min_idx_g3_diff4]
plt.axvline(x=h_opt_g3_diff2, color='r', linestyle=':', label=f'h_opt diff2 ≈ {h_opt_g3_diff2:.1e}')
plt.axvline(x=h_opt_g3_diff4, color='b', linestyle=':', label=f'h_opt diff4 ≈ {h_opt_g3_diff4:.1e}')
plt.xlabel('Шаг дифференцирования h')
plt.ylabel('Абсолютная погрешность')
plt.title('Погрешность для g3(x) при x=0.01')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('diff2_diff4_optimal_h_comparison.png')

plt.show()