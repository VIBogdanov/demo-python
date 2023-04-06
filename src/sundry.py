from collections import defaultdict
from itertools import accumulate


def find_intervals(
    elements: list[int],
    target: int = 0,
) -> list[tuple]:
    """
    Поиск в списке из чисел последовательного интервала(-ов) элементов, сумма которых равна
    искомому значению.
    Суть алгоритма выражена формулой: Sum1 - Sum2 = Target > Sum1 - Target = Sum2
      - Вычислить все суимы от начала до текущей позиции.
      - Для каждой суммы вычислить Sum - Target
      - Найти полученное значение в списке сумм
      - Если пара найдена, извлечь индексы и составить диапазон

    Args:
        elements (list[int]): Список неупорядоченных целых чисел, включая отрицательные значения.

        target (int): Искомое целое число, для которого ищется сумма элементов списка.

    Returns:
        list[tuple]: Результирующий список диапазонов элементов. Диапазоны задаются в виде
        кортежей пары целых чисел, обозначающих индексы элементов списка, включая начальный и
        конечный индексы. Если ни один диапазон не найден, возвращается пустой список.

    Example:
        >>> find_intervals([1, -3, 4, 5], 9)
        [(2, 3)]

        >>> find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0)
        [(0, 1), (4, 6), (8, 9), (9, 10)]
    """
    sum_dict: defaultdict = defaultdict(list[int])
    result_list: list[tuple] = list()

    try:
        target = int(target)
    except (ValueError, TypeError):
        return []

    # Суммируем элементы списка по нарастающей и сохраняем в словаре
    # список индексов для каждой суммы. В качестве ключа - сама сумма.
    for id_to, sum_accum in enumerate(accumulate(elements)):
        # Если на очередной итерации полученная сумма равна искомому значению,
        # заносим диапазон от 0 до текущей позиции в результирующий список.
        if sum_accum == target:
            result_list.append((0, id_to))
        # Ищем пару из уже вычисленных ранее сумм для значения (Sum - Target).
        # Если пара найдена, извлекаем индексы и формируем результирующие диапазоны.
        for id_from in sum_dict.get((sum_accum - target), []):
            result_list.append((id_from + 1, id_to))
        # Сохраняем очередную сумму и ее индекс в словаре, где ключ - сама сумма.
        sum_dict[sum_accum].append(id_to)

    return result_list


if __name__ == "__main__":
    #  result = find_intervals([1, -3, 4, 5], 1.2)
    # print(result)
    pass
