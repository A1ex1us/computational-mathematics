import numpy as np
import matplotlib.pyplot as plt

# Функция численного дифференцирования
def diff2(x_0, h, f):
    return (f(x_0 + h) - f(x_0 - h)) / (2 * h)

# Определение функций g1 и g3
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
h_values = np.logspace(-16, 0, 100)
exact_g1 = g1_prime(x0_g1)  # 4 * e^3 ≈ 80.342
exact_g3 = g3_prime(x0_g3)  # -10000 * π ≈ -31415.9265

# Вычисление погрешностей
errors_g1 = []
errors_g3 = []
for h in h_values:
    num_g1 = diff2(x0_g1, h, g1)
    num_g3 = diff2(x0_g3, h, g3)
    errors_g1.append(np.abs(num_g1 - exact_g1))
    errors_g3.append(np.abs(num_g3 - exact_g3))

# Построение графиков
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_g1, label='Погрешность для g1(x) при x=3')
plt.loglog(h_values, errors_g3, label='Погрешность для g3(x) при x=0.01')
plt.xlabel('Шаг дифференцирования h')
plt.ylabel('Абсолютная погрешность')
plt.title('Зависимость погрешности численного дифференцирования от h')
plt.grid(True)
plt.legend()
plt.savefig('error_plot.png')
plt.show()