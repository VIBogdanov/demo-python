from collections import Counter, defaultdict, deque
from collections.abc import Collection, Generator, Iterable
from functools import reduce
from itertools import chain, groupby, permutations
from math import prod
from typing import NamedTuple, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------------
def get_number_permutations(source_list: Iterable[T], target_list: Iterable[T]) -> int:
    """
    Подсчитывает минимальное количество перестановок, которое необходимо произвести для того,
    чтобы из исходного списка source_list получить целевой список target_list. При этом порядок
    следования и непрерывность списков не имеют значения.
    Например для списков:
    [10, 31, 15, 22, 14, 17, 16]
    [16, 22, 14, 10, 31, 15, 17]
    Требуется выполнить три перестановки для приведения списков в идентичное состояние.

    Args:
        source_list (Iterable[T]): Исходный список

        target_list (Iterable[T]): Целевой список

    Returns:
        int: Минимальное количество перестановок
    """
    # формируем список из номеров позиций для каждого значения из целевого списка.
    # Само значение является ключом.
    target_index: dict[T, int] = {n: i for i, n in enumerate(target_list)}
    # Генератор, который формирует номер позиции, на которую нужно переставить значение из исходного списка.
    source_index_generator: Generator[int, None, None] = (
        target_index[source_item] for source_item in source_list
    )
    count: int = 0
    # Получаем целевой номер позиции для первого значения из исходного списка
    prev_item = next(source_index_generator)
    # Попарно сравниваем целевые номера позиций для значений исходного списка.
    # Если номера позиций не по возрастанию, то требуется перестановка
    for next_item in source_index_generator:
        if prev_item > next_item:
            count += 1
        else:
            prev_item = next_item

    return count


# --------------------------------------------------------------------------------
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
        for mult_val in (
            (a_col for a_col in a_str if min_val <= a_col <= max_val)
            for a_str in matrix
        )
    ]


# --------------------------------------------------------------------------------
def count_items(
    data_list: Iterable[T],
    target: str,
    *,
    operation: str = "Total",
) -> int | float | None:
    """
    Функция подсчета количества нахождений заданного элемента в списке (по-умолчанию).
    При этом список может содержать неотсортированные и разнотипные элементы.
    Кроме общего количества элементов, возможно получить минимальный или максимальный
    размер группы или подсчитать количество групп.

    Args:
        data_list (list): Список неупорядоченных разнотипных элементов.

        target (str): Имя элемента в виде строки, для которого выполняется подсчет.

        operation (str, optional): Вид результирующего подсчета. Defaults to "Total".

    Returns:
        (int | float | None): Результат подсчета в зависимости от заданного вида.

    Example:
    >>> data = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 'a', 'a', 'a']
        count_items(data, '0')
        9
        count_items(data, '0', 'min')
        1
        count_items(data, '0', 'max')
        5
        count_items(data, '0', 'count')
        4

    """
    # словарь группирующий элементы из data_list
    dict_data: defaultdict = defaultdict(list)
    # словарь с перечнем операций над элементами групп
    dict_operations: dict = {
        # общее количество элементов во всех группах
        "total": lambda: sum(dict_data.get(target, [0])),
        # минимальный размер группы
        "min": lambda: min(dict_data.get(target, [0])),
        # максимальный размер группы
        "max": lambda: max(dict_data.get(target, [0])),
        # количество групп
        "count": lambda: len(dict_data.get(target, [0])),
        # ... возможно дальнейшее расширение списка операций
    }
    # список ключей из элементов и количество элементов в каждой группе
    _groups_list = ((str(k), len(tuple(v))) for k, v in groupby(data_list))
    for k, v in _groups_list:
        dict_data[k].append(v)
    # подсчитываем значение в зависимости от запрошенной операции
    # если вид операции задан некорректно, возвращается None
    return dict_operations.get(operation.lower(), lambda: None)()


# -------------------------------------------------------------------------------------------------
def get_pagebook_number(pages: int, count: int, digits: Iterable[int]) -> int:
    """
    Олимпиадная задача. Необходимо определить наибольший номер страницы X книги,
    с которой нужно начать читать книгу, чтобы ровно 'count' номеров страниц, начиная
    со страницы X и до последней страницей 'pages', заканчивались на цифры из списка  'digits'.

    Например:
    - вызов get_pagebook_number(1000000000000, 1234, [5,6]) вернет 999999993835
    - вызов get_pagebook_number(27, 2, [8,0]) вернет 18
    - вызов get_pagebook_number(20, 5, [4,7]) вернет 0

    Args:
        pages (int): Количество страниц в книге

        count (int): Количество страниц заканчивающихся на цифры из списка digits

        digits (Iterable[int]): Список цифр, на которые должны заканчиваться искомые страницы

    Returns:
        int: Номер искомой страницы или 0 в случае безуспешного поиска
    """
    len_lastdig: int = len(digits)
    if (count <= 0) and (pages < count) and (len_lastdig) == 0:
        return 0

    # Создаем копию и попутно удаляем дубликаты
    try:
        last_digits: list[int] = list(set(digits))
    except (ValueError, TypeError):
        return 0

    # Формируем список с ближайшими меньшими числами, оканчивающиеся на цифры из списка digits
    for i in range(len_lastdig):
        last_digits[i] = (pages - last_digits[i]) // 10 * 10 + last_digits[i]

    # Полученный список обязательно должен быть отсортирован в обратном порядке
    last_digits.sort(reverse=True)
    # Вычисляем позицию числа, которое соответствует смещению count
    idx: int = (count % len_lastdig) - 1
    # Т.к. последующая последняя цифра повторяется через 10,
    # вычисляем множитель с учетом уже вычисленных значений
    multiplier: int = (count - 1) // len_lastdig
    return last_digits[idx] - (multiplier * 10)


# -------------------------------------------------------------------------------------------------
def get_combination_numbers(digits: Collection[int]) -> list[tuple[int, ...]]:
    """
    Сформировать все возможные уникальные наборы чисел из указанных цифр.
    Цифры можно использовать не более одного раза в каждой комбинации.
    При этом числа не могут содержать в начале 0 (кроме самого нуля).

    Args:
        digits (list[int]): Список заданных цифр

    Returns:
        list[int]: Список уникальных комбинаций
    """

    # Класс данных для удобства загрузки-выгрузки данных в очередь
    class ComboNums(NamedTuple):
        # Список двух-, трех-  т.д. значных цифр, составленных из цифр исходного списка
        Numbers_List: list[int]
        # Оставшиеся от исходного списка цифры после формирования списка чисел
        Digits_List: Collection[int]

    # Результирующий список как множество, дабы исключить дубликаты
    results = set()
    # Предварительно в результирующий список сохраняем все комбинации из цифр
    # исходного списка (состоящие из одной цифры) и удаляем дубли
    results = set(permutations(digits))

    # Очередь потребуется для хранения списков, требующих обработки
    query_buff: deque = deque()
    # Перед запуском цикла обработки загружаем в очередь исходный список
    query_buff.append(ComboNums(list(), digits))

    while query_buff:
        # Пока в очереди есть что обрабатывать выгружаем список для обработки
        combo_numbers: ComboNums = query_buff.pop()
        # Формируем генератор, который из цифр списка составляет двухзначные, трехзначные и т.д. числа
        # Т.к. числа, состоящие из одной цифры, уже обработаны, начинаем с двухзначных чисел
        gen_digits_list: Generator[tuple[int, ...], None, None] = (
            digit
            for perms in (
                set(permutations(combo_numbers.Digits_List, i))
                for i in range(2, len(combo_numbers.Digits_List) + 1)
            )
            for digit in perms
            if digit[0] != 0  # Исключаем числа начинающиеся с нуля
        )
        # Перебираем все полученные двухзначные, трехзначные и т.д. числа
        for selected_digits in gen_digits_list:
            # Удаляем из списка цифры, из которых состоят двухзначных, трехзначных и т.д. числа.
            # Например, для числа 12 из исходного списка удаляем цифры 1 и 2
            digits_count = Counter(combo_numbers.Digits_List)
            digits_count.subtract(selected_digits)
            # Из отобранных цифр (в нашем примере 1 и 2) формируем число 12
            num: int = reduce(
                lambda dig_prev, dig_next: 10 * dig_prev + dig_next, selected_digits
            )
            # В результирующий список записываем все возможные комбинации числа 12 и оставшихся цифр
            results.update(
                set(
                    permutations(
                        combo_numbers.Numbers_List
                        + [num]
                        + list(digits_count.elements())
                    )
                )
            )
            # Если количество оставшихся цифр в списке 2 и более, то есть возможность
            # сформировать комбинации из трехзначных и более чисел из оставшихся цифр.
            if digits_count.total() > 1:
                query_buff.append(
                    ComboNums(
                        combo_numbers.Numbers_List + [num],
                        list(digits_count.elements()),
                    )
                )
    # Формируем список результатов и сортируем его для удобства отображения
    list_results = list(results)
    list_results.sort()
    return list_results


if __name__ == "__main__":
    pass
