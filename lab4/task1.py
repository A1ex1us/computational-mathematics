import numpy as np


def lu(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)  # Инициализация L как единичной матрицы
    U = A.copy()  # Инициализация U копией матрицы A

    for i in range(n - 1):  # Для каждого столбца i
        for j in range(i + 1, n):  # Для строк j ниже диагонали
            if U[i, i] == 0:
                raise ValueError("Нулевой элемент на диагонали. Требуется выбор главного элемента.")
            L[j, i] = U[j, i] / U[i, i]  # Сохраняем множитель в L
            U[j, i:] -= L[j, i] * U[i, i:]  # Обновляем строку j матрицы U

    return L, U