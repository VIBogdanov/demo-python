from itertools import chain
from math import prod


def mult_matrix(
    matrix: list[list[float | int]],
    min_val: float | int | None = None,
    max_val: float | int | None = None,
    default_val: float | int = 0.0,
) -> list[float]:
    """
    Функция получения вектора из двумерной матрицы путем перемножения
    значений в каждой строке, при этом значения должны входить в диапазон
    от min_val до max_val.

    Args:
        matrix (list[list[float | int]]): двумерная числовая матрица

        min_val (float | int | None, optional): Минимальная граница диапазона. Defaults to None.

        max_val (float | int | None, optional): Максимальная граница диапазона. Defaults to None.

        default_val (float | int, optional): Генерируемое значение в случае невозможности
        выполнить перемножение. Defaults to 0.00.

    Returns:
        list[float]: Список значений как результат построчного перемножения двумерной матрицы.
        Количество элементов в списке соответствует количеству строк в матрице.

    Example:
        >>> matrix = [
                [1.192, 1.192, 2.255, 0.011, 2.167],
                [1.192, 1.192, 2.255, 0.011, 2.167],
                [2.255, 2.255, 1.734, 0.109, 5.810],
                [0.011, 0.011, 0.109, 0.420, 1.081],
                [2.167, 2.167, 5.810, 1.081, 0.191]
            ]
            mult_matrix(matrix,2,10)
            [4.887, 4.887, 29.544, 0, 27.283]

    """
    # Если минимальное ограничение не задано, задаем как минимальное значение из матрицы
    if min_val is None:
        min_val = min(chain.from_iterable(matrix))
    # Если максимальное ограничение не задано, задаем как максимальное значение из матрицы
    if max_val is None:
        max_val = max(chain.from_iterable(matrix))
    # Корректируем ошибочные ограничения
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    # Читаем построчно значения из матрицы и поэлементно перемножаем для каждой строки
    return [
        # если ни один элемент из строки не удовлетворяет ограничениям, возвращаем значение по-умолчанию
        # иначе перемножаем отфильтрованные значения
        round(prod(mult_val), 3) if mult_val else default_val
        for mult_val in ((a_col for a_col in a_str if min_val <= a_col <= max_val) for a_str in matrix)
    ]


if __name__ == "__main__":
    pass
