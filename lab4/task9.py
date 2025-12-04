import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def lu(A, permute=False):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)  # Матрица перестановок
    U = A.copy()

    for i in range(n-1):
        if permute:
            # Поиск индекса максимального элемента в столбце i, начиная с i-й строки
            max_row = i + np.argmax(np.abs(U[i:, i]))
            if max_row != i:
                # Перестановка строк в U
                U[[i, max_row], :] = U[[max_row, i], :]
                # Перестановка строк в L (только уже вычисленные элементы)
                L[[i, max_row], :i] = L[[max_row, i], :i]
                # Обновление матрицы перестановок
                P[[i, max_row], :] = P[[max_row, i], :]

        if U[i, i] == 0:
            raise ValueError("Нулевой диагональный элемент. Требуется выбор главного элемента (permute=True).")

        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return L, U, P


def solve(L, U, P, b):
    """
    Решает систему Ax = b, используя LU-разложение с матрицей перестановок P (PA = LU).

    Параметры:
        L (np.ndarray): Нижняя треугольная матрица с единицами на диагонали.
        U (np.ndarray): Верхняя треугольная матрица.
        P (np.ndarray): Матрица перестановок.
        b (np.ndarray): Вектор правой части системы.

    Возвращает:
        np.ndarray: Вектор решения x.
    """
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)
    b = np.array(b, dtype=float).flatten()

    # Применяем перестановку к вектору b: P b = P @ b
    b_permuted = P @ b

    # Прямой ход: Решаем Ly = Pb
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

    # Обратный ход: Решаем Ux = y
    for i in reversed(range(n)):
        if U[i, i] == 0:
            raise ValueError("Деление на ноль: диагональный элемент U равен нулю.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# Матрица A1 и вектор b1 из задания
A1 = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)
b1 = np.array([4, 1, -3, 4])

# Вычисляем LU-разложение A1
L1, U1, P1 = lu(A1)  # Добавлена переменная P1
print("L1:\n", L1.round(3))
print("U1:\n", U1.round(3))
print("P1:\n", P1.round(3))  # Вывод матрицы перестановок

# Решаем систему
x1 = solve(L1, U1, P1, b1)  # Добавлен аргумент P1
print("\nРешение x1:", x1.round(3))
print("Проверка A1 * x1:", np.dot(A1, x1).round(3))

# Матрица A2 и вектор b2 из задания
A2 = np.array([
    [3, 1, -3],
    [6, 2, 5],
    [1, 4, -3]
], dtype=float)
b2 = np.array([-16, 12, -39])

# LU-разложение с выбором главного элемента (библиотечная реализация)
lu_A2, piv = lu_factor(A2)
x2 = lu_solve((lu_A2, piv), b2)
print("\nРешение x2:", x2.round(3))
print("Проверка A2 * x2:", np.dot(A2, x2).round(3))


# Определение матрицы A2 и вектора b2
A2 = np.array([
    [3, 1, -3],
    [6, 2, 5],
    [1, 4, -3]
], dtype=float)

b2 = np.array([-16, 12, -39])

# LU-разложение с выбором главного элемента
L, U, P = lu(A2, permute=True)

# Решение системы
x_tilde = solve(L, U, P, b2)

# Проверка корректности
result = np.dot(A2, x_tilde)
is_correct = np.allclose(result, b2, atol=1e-5)

print("Решение x_tilde:", x_tilde.round(3))
print("Проверка A2 * x_tilde ≈ b2:", is_correct)

# Исходные данные
A2_original = np.array([
    [3, 1, -3],
    [6, 2, 5],
    [1, 4, -3]
], dtype=float)

b2_original = np.array([-16, 12, -39], dtype=float)
exact_solution = np.array([1, -7, 4], dtype=float)

# Диапазон значений p
p_values = np.arange(0, 13)

# Массивы для сохранения ошибок
errors_without_pivoting = []
errors_with_pivoting = []

for p in p_values:
    epsilon = 10.0 ** (-p)

    # Модификация матрицы A2 и вектора b2
    A2_modified = A2_original.copy()
    A2_modified[0, 0] += epsilon  # Добавляем 10^{-p} к a11
    b2_modified = b2_original.copy()
    b2_modified[0] += epsilon  # Добавляем 10^{-p} к b1

    # Случай 1: Без выбора главного элемента
    try:
        L, U, P = lu(A2_modified, permute=False)
        x_without_pivot = solve(L, U, P, b2_modified)
        error = np.linalg.norm(x_without_pivot - exact_solution)
        errors_without_pivoting.append(error)
    except ValueError as e:
        errors_without_pivoting.append(np.nan)  # Ошибка при нулевом диагональном элементе

    # Случай 2: С выбором главного элемента
    L_piv, U_piv, P_piv = lu(A2_modified, permute=True)
    x_with_pivot = solve(L_piv, U_piv, P_piv, b2_modified)
    error = np.linalg.norm(x_with_pivot - exact_solution)
    errors_with_pivoting.append(error)

# Вывод результатов
print("p\tБез выбора\tС выбором")
for i, p in enumerate(p_values):
    print(f"{p}\t{errors_without_pivoting[i]:.2e}\t{errors_with_pivoting[i]:.2e}")

