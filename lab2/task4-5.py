import numpy as np
import matplotlib.pyplot as plt

# Функция составной формулы Симпсона
def composite_simpson(a, b, n, f):
    if n % 2 == 0:
        raise ValueError("Число узлов n должно быть нечётным.")
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    fx = f(x)
    integral = fx[0] + fx[-1]
    for i in range(1, n-1):
        if i % 2 == 1:
            integral += 4 * fx[i]
        else:
            integral += 2 * fx[i]
    return (h / 3) * integral

# Определение функций
def g2(x):
    return x**2 * np.sin(3*x)

def g3(x):
    return np.sin(np.pi / x)

# Параметры
pi = np.pi
epsilon = 0.01
exact_I2 = (1/3)*pi**2 - 4/27  # Точное значение для g2
exact_I3_approx = composite_simpson(epsilon, 1, 100001, g3)  # Приблизительное точное значение для g3

# Сетка для n
n_list = np.round(np.logspace(np.log10(3), np.log10(9999), 20)).astype(int)
n_list = [n if n % 2 == 1 else n + 1 for n in n_list]  # Обеспечиваем нечётность

# Вычисление погрешностей для g2
errors_g2 = []
h_list_g2 = []
for n in n_list:
    h = (pi - 0)/(n-1)
    I_num = composite_simpson(0, pi, n, g2)
    error = np.abs(I_num - exact_I2)
    errors_g2.append(error)
    h_list_g2.append(h)

# Вычисление погрешностей для g3
errors_g3 = []
h_list_g3 = []
for n in n_list:
    h = (1 - epsilon)/(n-1)
    I_num = composite_simpson(epsilon, 1, n, g3)
    error = np.abs(I_num - exact_I3_approx)
    errors_g3.append(error)
    h_list_g3.append(h)

# Построение графиков
plt.figure(figsize=(12, 6))

# График для g2
plt.subplot(1, 2, 1)
plt.loglog(h_list_g2, errors_g2, 'b-o', label='Погрешность')
plt.xlabel('Шаг интегрирования h')
plt.ylabel('Абсолютная погрешность')
plt.title('Погрешность для g2(x) на [0, π]')
plt.grid(True)
plt.legend()

# График для g3
plt.subplot(1, 2, 2)
plt.loglog(h_list_g3, errors_g3, 'r-o', label='Погрешность')
plt.xlabel('Шаг интегрирования h')
plt.ylabel('Абсолютная погрешность')
plt.title('Погрешность для g3(x) на [0.01, 1]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('error_plots.png')
plt.show()