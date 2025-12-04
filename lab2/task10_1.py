import numpy as np
import matplotlib.pyplot as plt

# Функция diff4
def diff4(x_0, h, f):
    return (1/(12*h)) * (-f(x_0 - 2*h) + 8*f(x_0 - h) - 8*f(x_0 + h) + f(x_0 + 2*h))

# Функция и её производная
def f(x):
    return np.sin(x)

def f_prime(x):
    return np.cos(x)

# Параметры
x0 = 1.0
exact = f_prime(x0)  # cos(1) ≈ 0.540302
h_values = np.logspace(-6, -1, 50)
errors = []

# Вычисление погрешностей
for h in h_values:
    num = diff4(x0, h, f)
    error = np.abs(num - exact)
    errors.append(error)

# Построение log–log графика
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors, 'b-o', label='Погрешность diff4')
# Аппроксимация наклона
mask = (h_values >= 1e-4) & (h_values <= 1e-2)
log_h = np.log10(h_values[mask])
log_errors = np.log10(np.array(errors)[mask])
coeff = np.polyfit(log_h, log_errors, 1)
slope = coeff[0]
plt.loglog(h_values[mask], 10**(coeff[1]) * h_values[mask]**slope, 'r--', label=f'Наклон = {slope:.2f}')
plt.xlabel('Шаг дифференцирования h')
plt.ylabel('Абсолютная погрешность')
plt.title('Погрешность diff4 для f(x) = sin(x) при x=1')
plt.grid(True)
plt.legend()
plt.savefig('diff4_order_loglog.png')
plt.show()