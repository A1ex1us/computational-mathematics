import numpy as np
import math


def gauss_quad5(f):
    # Узлы и веса для 5-точечной схемы (фактически 3 точки)
    nodes = [-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)]
    weights = [5 / 9, 8 / 9, 5 / 9]

    # Вычисление взвешенной суммы
    result = 0.0
    for i in range(3):
        result += weights[i] * f(nodes[i])
    return result


def integrate_with_transformation(func, lower, upper):
    def transformed(t):
        # Преобразование координат для общего интервала
        midpoint = (upper + lower) / 2
        half_range = (upper - lower) / 2
        x = midpoint + half_range * t
        return func(x) * half_range

    return gauss_quad5(transformed)


def compute_antiderivative(poly):
    # Вычисление первообразной через коэффициенты
    coeffs = poly.coeffs
    n = len(coeffs)
    new_coeffs = []

    # Интегрирование каждого члена
    for idx, coeff in enumerate(coeffs):
        exponent = n - idx - 1
        new_coeffs.append(coeff / (exponent + 1))
    new_coeffs.append(0)  # Константа интегрирования

    return np.poly1d(new_coeffs)


def format_polynomial(coeffs):
    # Форматирование полинома в читаемую строку
    n = len(coeffs)
    terms = []

    for i in range(n):
        power = n - i - 1
        coeff = coeffs[i]
        if abs(coeff) < 1e-10:
            continue

        if power == 0:
            term = f"{coeff:.4f}"
        elif power == 1:
            term = f"{coeff:.4f}x"
        else:
            term = f"{coeff:.4f}x^{power}"
        terms.append(term)

    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    # Параметры анализа
    degrees = [0, 1, 2, 3, 4, 5, 6]
    polynomials = []
    integration_bounds = (0, 2)

    # Генерация случайных полиномов
    for deg in degrees:
        coeffs = np.random.standard_normal(deg + 1)
        polynomials.append(np.poly1d(coeffs))

    # Вывод сгенерированных полиномов
    print("Случайно сгенерированные полиномы:")
    for deg, poly in zip(degrees, polynomials):
        print(f"Степень {deg}: P(x) = {format_polynomial(poly.coeffs)}")

    # Заголовок таблицы результатов
    header = ["Степень", "Точное значение", "Численный результат", "Погрешность"]
    print("\n" + " | ".join(f"{h:^25}" for h in header))
    print("-" * 110)

    # Вычисление и сравнение интегралов
    for deg, poly in zip(degrees, polynomials):
        # Точное значение интеграла
        F = compute_antiderivative(poly)
        exact_val = F(integration_bounds[1]) - F(integration_bounds[0])

        # Численное интегрирование
        num_val = integrate_with_transformation(poly, *integration_bounds)

        # Расчёт погрешности
        error = abs(exact_val - num_val)

        # Форматированный вывод
        print(f"{deg:^25} | {exact_val:^25.8f} | {num_val:^25.8f} | {error:^25.8f}")