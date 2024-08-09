from array import array
from collections import Counter, defaultdict, deque
from collections.abc import Generator, Iterable, Iterator
from functools import reduce
from itertools import chain, groupby, permutations
from math import prod
from typing import Any, TypeAlias, TypeVar

from assistools import ilen

T = TypeVar("T")
TIntNone: TypeAlias = int | None


# --------------------------------------------------------------------------------
def get_min_permutations(source_list: Iterable[T], target_list: Iterable[T]) -> int:
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
def get_combination_numbers(digits: Iterable[int]) -> list[tuple[int, ...]]:
    """
    Сформировать все возможные уникальные наборы чисел из указанных цифр.
    Цифры можно использовать не более одного раза в каждой комбинации.
    При этом числа не могут содержать в начале 0 (кроме самого нуля).

    Args:
        digits: Список заданных цифр

    Returns:
        list[int]: Список уникальных комбинаций
    """

    # Предварительно в результирующий список сохраняем все комбинации
    # из одиночных цифр исходного списка и удаляем дубли
    results = set(permutations(digits))
    range_size = ilen(digits) + 1
    # Формируем генератор, который из цифр списка составляет двухзначные, трехзначные и т.д. числа
    # Т.к. числа, состоящие из одной цифры, уже обработаны, начинаем с двухзначных чисел
    gen_digits_list: Generator[tuple[int, ...], None, None] = (
        combo_digits
        for perms in (set(permutations(digits, i)) for i in range(2, range_size))
        for combo_digits in perms
        if combo_digits[0] != 0  # Исключаем числа начинающиеся с нуля
    )
    # Перебираем все полученные двухзначные, трехзначные и т.д. наборы цифр
    for selected_digits in gen_digits_list:
        # Из отобранных цифр (в нашем примере 1 и 2) формируем число 12
        num: int = reduce(
            lambda dig_prev, dig_next: 10 * dig_prev + dig_next, selected_digits
        )
        # Удаляем из списка цифры, из которых состоят двухзначных, трехзначных и т.д. числа.
        # Например, для числа 12 из исходного списка удаляем цифры 1 и 2
        digits_count = Counter(digits)
        digits_count.subtract(selected_digits)
        # В результирующий список записываем все возможные комбинации числа 12 и оставшихся цифр
        results.update(set(permutations(chain((num,), digits_count.elements()))))

    # Формируем список результатов и сортируем его для удобства отображения
    list_results = list(results)
    list_results.sort()
    return list_results


# -------------------------------------------------------------------------------------------------
def closest_amount(
    numbers: Iterable[int], target: int
) -> tuple[int, list[tuple[int, ...]]]:
    """
    Получить число, максимально близкое к числу X, из суммы неотрицательных чисел массива,
    при условии, что числа из массива могут повторяться.

    Args:
        numbers: Массив чисел

        target: Целевое число

    Returns:
        tuple[int, list]: Кортеж из искомого числа и списка(-ов) наборов чисел,
        сумма которых равна искомому числу.
    """
    # Искомое число и списки чисел, суммы которых равны искомому числу
    max_sum: int = 0
    # Используем set, дабы исключить дубли
    max_sum_numbers: set[tuple[int, ...]] = set()

    # Очередь из промежуточных накопительных сумм и списков чисел, из которых состоят эти суммы
    query_buff: deque[tuple[int, list[int]]] = deque()
    # Стартуем с нуля
    query_buff.append((0, []))

    while query_buff:
        # Вынимаем из очереди очередную промежуточную сумму
        # и поочередно суммируем ее с числами из входного массива.
        current_sum, current_numbers = query_buff.popleft()
        # Для перебора чисел из входного массива используем генератор, который отфильтровывает
        # отрицательные числа и превышение целевого числа. Генератор формирует кортеж из
        # следующей суммы и набора чисел, ее составляющих.
        for next_sum, next_numbers in (
            (
                (current_sum + number),
                # Сортировка позволяет избежать дублирования списков при добавлении в max_sum_numbers
                sorted(current_numbers + [number]),
            )
            for number in numbers
            if (lambda _number: _number > 0 and (current_sum + _number) <= target)(
                number
            )
        ):
            # Если очередная полученная сумма больше ранее вычисленной максимальной суммы,
            # обновляем максимальную сумму и список чисел, которые ее формируют.
            if next_sum > max_sum:
                max_sum = next_sum
                max_sum_numbers.clear()
                max_sum_numbers.add(tuple(next_numbers))
            # Одна и та же сумма, может быть получена различными комбинациями чисел из входного массива.
            elif next_sum == max_sum:
                # Т.к. max_sum_numbers - это set, то совпадающие списки чисел отфильтровываются
                max_sum_numbers.add(tuple(next_numbers))
            # Добавляем в очередь очередную сумму со списком чисел для дальнейшей обработки.
            if (next_sum, next_numbers) not in query_buff:
                query_buff.append((next_sum, next_numbers))

    return (max_sum, list(max_sum_numbers))


# -------------------------------------------------------------------------------------------------
def get_minmax_prod(iterable: Iterable[int]) -> tuple[TIntNone, TIntNone]:
    """
    Находит две пары множителей в массиве чисел, дабы получить минимально возможное
    и максимально возможное произведение. Допускаются отрицательные значения и ноль.

    Details: Попытка реализовать максимально обобщенный вариант без индексирования,
    без сортировки, без вычисления размера данных и без изменения исходных данных.
    Используется только итератор.

    Args:
        iterable: Набор чисел.

    Returns:
        tuple(min, max): Пара минимального и максимального значений произведения.
    """
    result: tuple[TIntNone, TIntNone] = (None, None)
    # Получаем итератор для однократного прохождения по элементам данных.
    # По производительности сравнимо с сортировкой, но при этом не модифицирует исходные данные.
    it_elements: Iterator[int] = iter(iterable)

    # Инициализируем первыми значениями исходных данных.
    try:
        min1 = max1 = next(it_elements)
    except StopIteration:
        return result  # Если список исходных данных пуст

    try:
        min2 = max2 = next(it_elements)
    except StopIteration:
        return (min1, max1)  # Список исходных данных состоит из одного значения

    # Важно изначально инициализировать min и max корректными относительными значениями
    if min2 < min1:
        min1, min2 = min2, min1

    if max1 < max2:
        max1, max2 = max2, max1

    for elm in it_elements:
        # Вычисляем первые два минимальные значения
        if elm < min1:
            min1, min2 = elm, min1
        elif elm < min2:
            min2 = elm
        #  и последние два максимальные значения
        if max1 < elm:
            max1, max2 = elm, max1
        elif max2 < elm:
            max2 = elm
    # Данные произведения потребуются далее. Для читабельности кода.
    min_prod = min1 * min2
    max_prod = max1 * max2

    match (min1 < 0, max1 < 0):
        case (True, True):  # Все числа отрицательные
            result = (max_prod, min_prod)
        case (True, False):  # Часть чисел отрицательные
            result = (min1 * max1, min_prod if (max_prod < min_prod) else max_prod)
        case (False, False):  # Все числа неотрицательные (включая ноль)
            result = (min_prod, max_prod)
    return result


# -------------------------------------------------------------------------------------------------
def get_incremental_list(digits: Iterable[int]) -> tuple[int, list[int]]:
    """Из заданного набора целых чисел получить список возрастающих чисел
    за минимальное количество изменений исходного списка. Возможны как
    положительные, так и отрицательные значения, включая ноль. Сортировка не требуется.

    Example:
    get_incremental_list([1, 7, 3, 3]) -> (2, [1, 2, 3, 4])
    get_incremental_list([3, 2, 1]) -> (0, [3, 2, 1])
    get_incremental_list([-2, 0, 4]) -> (1, [-2, 0, -1])

    Args:
        digits (Iterable[int]): Заданный список целых чисел.

    Returns:
        tuple[int,list[int]]: Количество изменений и список возрастающих чисел.
    """
    # Значение, с которого начинается отсчет
    start = min(digits)
    # Значение, которым должен заканчиваться результирующий список
    end = (ilen(digits) + start) - 1
    # Выясняем сколько и какие числа нужно подставить в исходный список,
    # чтобы получить возрастающий список. Первый set строит полную
    # последовательность возрастающих чисел, второй set удаляет те числа,
    # которые уже присутствуют в исходном списке.
    missing_digits = set(range(start, end + 1)) - set(digits)
    # Минимальное количество требуемых изменений
    cnt = len(missing_digits)
    result = list()
    # Проверяем каждое значение исходного списка: если значение не входит
    # в результирующий диапазон или продублировано, заменяем его на число
    # из списка для подстановки
    for val in digits:
        if val > end or val in result:
            val = missing_digits.pop()
        result.append(val)
    return (cnt, result)


# -------------------------------------------------------------------------------------------------
def get_word_palindrom(chars: str) -> str:
    """Из заданного набора символов сформировать палиндром.

    Args:
        chars - Список символов.

    Returns:
        str - Палиндром. Если сформировать палиндром не удалось, возвращается пустая строка.
    """
    # Массив для аккумулирования кандидатов для символа-разделителя между половинами палиндрома
    midl_candidate = array("u")

    # Внутренняя функция генератор для формирования символов, входящих в палиндром
    def gwp(chrs: str) -> Generator[str, Any, None]:
        # Подсчитываем количество символов в заданном наборе и запускаем цикл их перебора
        for _char, _count in Counter(chrs).items():
            # Если количество символа нечетное, то это потенциальный символ-разделитель
            if _count & 1:
                midl_candidate.append(_char)
            # Возвращаем только символы, у которых количество пар одна и более
            if pair_count := (_count >> 1):
                yield str(_char * pair_count)

    # Формируем левую половину палиндрома
    half_palindrom: str = "".join(sorted(gwp(chars)))
    if len(half_palindrom):
        # Определяем символ-разделитель как лексикографически минимальный
        midl_symbol: str = min(midl_candidate) if len(midl_candidate) else ""
        # Собираем результирующий палиндром
        return "".join((half_palindrom, midl_symbol, half_palindrom[::-1]))
    return ""


# -------------------------------------------------------------------------------------------------
def main():
    print("\n- Сформировать все возможные уникальные наборы чисел из указанных цифр.")
    print(
        f" get_combination_numbers([0, 2, 7]) -> {get_combination_numbers([0, 2, 7])}"
    )

    print(
        "\n- Минимальное количество перестановок, которое необходимо произвести для выравнивания списков."
    )
    print(
        f" get_number_permutations([10, 31, 15, 22, 14, 17, 16], [16, 22, 14, 10, 31, 15, 17])) -> {get_min_permutations([10, 31, 15, 22, 14, 17, 16], [16, 22, 14, 10, 31, 15, 17])}"
    )

    print("\n- Олимпиадная задача. См. описание в puzzles.py.")
    print(f" get_pagebook_number(27, 2, [8,0]) -> {get_pagebook_number(27, 2, [8,0])}")

    print("\n- Получить число, максимально близкое к числу X, из суммы чисел массива.")
    print(f" closest_amount([20, 30, 38], 112) -> {closest_amount([20, 30, 38], 112)}")

    print(
        "\n- Находит минимальное и максимальное произведение пары чисел из списка значений."
    )
    print(f" get_minmax_prod([1, -2, 5, 4]) -> {get_minmax_prod([1, -2, 5, 4])}")

    print(
        "\n- Из заданного набора целых чисел получить список возрастающих чисел за минимальное количество изменений исходного списка."
    )
    print(
        f" get_incremental_list([1, 7, 3, 3]) -> {get_incremental_list([1, 7, 3, 3])}"
    )

    print("\n- Из заданного набора символов формирует слово-палиндром.")
    print(f" get_word_palindrom('bbaadcbb') -> {get_word_palindrom('bbaadcbb')}")

    print("\n")


if __name__ == "__main__":
    main()
