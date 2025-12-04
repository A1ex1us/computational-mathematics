import numpy as np

def solve(L, U, b):
    """
    Решает систему Ax = b, используя LU-разложение A = L*U.
    Возвращает вектор x.
    """
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)
    b = np.array(b, dtype=float).flatten()  # Преобразуем в одномерный массив

    # Прямой ход: Ly = b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Обратный ход: Ux = y
    for i in reversed(range(n)):
        if U[i, i] == 0:
            raise ValueError("Деление на ноль: диагональный элемент U равен нулю.")
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x