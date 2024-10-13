import array
import functools
import math
from collections import Counter, defaultdict, deque
from collections.abc import Generator, Iterable, Iterator
from enum import Enum
from itertools import chain, groupby, permutations
from typing import Literal, TypeAlias, TypeVar, cast

# Должно быть так: from .assistools import ilen
# Но это ограничивает независимый запуск файла puzzles.py, который в составе модуля
import demo

T = TypeVar("T")
TIntNone: TypeAlias = int | None
TNumber: TypeAlias = int | float


# --------------------------------------------------------------------------------
def get_min_permutations(source_list: Iterable[T], target_list: Iterable[T]) -> int:
    """
    Подсчитывает минимальное количество перестановок, которое необходимо произвести для того,
    чтобы из исходного списка source_list получить целевой список target_list. При этом порядок
    следования и непрерывность списков не имеют значения.

    Args:
        source_list (Iterable[T]): Исходный список

        target_list (Iterable[T]): Целевой список

    Returns:
        int: Минимальное количество перестановок

    Example:
        >>> source = [10, 31, 15, 22, 14, 17, 16]
            target = [16, 22, 14, 10, 31, 15, 17]
            get_min_permutations(source, target)
            3
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
    try:
        prev_index = next(source_index_generator)
    except StopIteration:
        return count  # список пуст
    # Попарно сравниваем целевые номера позиций для значений исходного списка.
    # Если номера позиций не по возрастанию, то требуется перестановка
    for next_index in source_index_generator:
        if prev_index > next_index:
            count += 1
        else:
            prev_index = next_index

    return count


# --------------------------------------------------------------------------------
def mult_matrix(
    matrix: list[list[TNumber]],
    min_val: TNumber | None = None,
    max_val: TNumber | None = None,
    default_val: TNumber = 0.0,
) -> list[float]:
    """
    Функция получения вектора из двумерной матрицы путем перемножения
    значений в каждой строке, при этом значения должны входить в диапазон
    от min_val до max_val.

    Args:
        matrix (list[list[float | int]]): двумерная числовая матрица

        min_val (float | int | None, optional): Минимальная граница диапазона. Defaults to None.

        max_val (float | int | None, optional): Максимальная граница диапазона. Defaults to None.

        default_val (float | int, optional): Генерируемое значение в случае невозможности \
        выполнить перемножение. Defaults to 0.00.

    Returns:
        (list[float]): Список значений как результат построчного перемножения двумерной матрицы.
        Количество элементов в списке соответствует количеству строк в матрице.

    Example:
        >>> matrix = [
            [1.192, 1.192, 2.255, 0.011, 2.167],
            [1.192, 1.192, 2.255, 0.011, 2.167],
            [2.255, 2.255, 1.734, 0.109, 5.810],
            [0.011, 0.011, 0.109, 0.420, 1.081],
            [2.167, 2.167, 5.810, 1.081, 0.191]
            ]
            mult_matrix(matrix, 2,10)
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
        round(math.prod(mult_val), 3) if mult_val else default_val
        for mult_val in (
            (a_col for a_col in a_str if min_val <= a_col <= max_val)
            for a_str in matrix
        )
    ]


# --------------------------------------------------------------------------------
TOperation: TypeAlias = Literal["Total", "Min", "Max", "Count"]


def count_items(
    data_list: Iterable,
    target: str,
    *,
    operation: TOperation = "Total",
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
    len_lastdig: int = demo.ilen(digits)
    if (count <= 0) and (pages < count) and (len_lastdig == 0):
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
        digits (Iterable[int])): Список заданных цифр

    Returns:
        (list[int]): Список уникальных комбинаций
    """

    # Предварительно в результирующий список сохраняем все комбинации
    # из одиночных цифр исходного списка и удаляем дубли
    results = set(permutations(digits))
    range_size = demo.ilen(digits) + 1
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
        num: int = functools.reduce(
            lambda dig_prev, dig_next: 10 * dig_prev + dig_next, selected_digits
        )
        # Удаляем из списка цифры, из которых состоят двухзначных, трехзначных и т.д. числа.
        # Например, для числа 12 из исходного списка удаляем цифры 1 и 2
        digits_count = Counter(digits)
        digits_count.subtract(selected_digits)
        # В результирующий список записываем все возможные комбинации числа 12 и оставшихся цифр
        results.update(set(permutations(chain((num,), digits_count.elements()))))

    # Формируем список результатов и сортируем его для удобства отображения
    return sorted(results)


# -------------------------------------------------------------------------------------------------
def closest_amount(
    numbers: Iterable[int], target: int
) -> tuple[int, list[tuple[int, ...]]]:
    """
    Получить число, максимально близкое к числу X, из суммы неотрицательных чисел массива,
    при условии, что числа из массива могут повторяться.

    Args:
        numbers (Iterable[int]): Массив чисел

        target (int): Целевое число

    Returns:
        (tuple[int, list]): Кортеж из искомого числа и списка(-ов) наборов чисел,
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
            # lambda использована исключительно в демонстрационных целях
            # проще просто:  if number > 0 and (current_sum + number) <= target
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
        iterable (Iterable[int]): Набор чисел.

    Returns:
        (tuple(int, int)): Пара минимального и максимального значений произведения.
    """
    result: tuple[TIntNone, TIntNone] = (None, None)
    # Получаем итератор для однократного прохода по элементам данных.
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
def get_incremental_list(digits: Iterable[int]) -> tuple[int, list[int], list]:
    """Из заданного набора целых чисел получить список возрастающих чисел
    за минимальное количество изменений исходного списка. Возможны как
    положительные, так и отрицательные значения, включая ноль. Сортировка не требуется.

    Example:
    get_incremental_list([1, 7, 3, 3]) -> (2, [1, 3], [1, 2, 3, 4])
    get_incremental_list([3, 2, 1]) -> (0, [], [3, 2, 1])
    get_incremental_list([-2, 0, 4]) -> (1, [2], [-2, 0, -1])

    Args:
        digits (Iterable): Заданный список целых чисел.

    Returns:
        (tuple[int,list,list]): Количество изменений, измененные позиции и список возрастающих чисел.
    """
    # Итератор позволит работать даже с генераторами.
    it_digits: Iterator[int] = iter(digits)
    try:
        # Начальное значение результирующего списка инициализируем
        # первым элементом входных данных
        begin: int = next(it_digits)
    except StopIteration:
        return (0, [], [])  # Если список исходных данных пуст

    result: list[int | None] = [begin]
    # Используем set, чтобы ускорить поиск по списку уже обработанных элементов
    # Дополнительные затраты памяти на set - это плата за скорость поиска
    # Если память важнее, вместо elements_used нужно использовать список result
    elements_used: set[int] = {begin}
    # Индексы позиций в списке, в которых произошла замена
    positions: list[int] = list()

    # Стартуем со второго элемента входных данных
    for dig in it_digits:
        # Дубликаты заменяем на None
        if dig in elements_used:
            result.append(None)
        else:
            result.append(dig)
            elements_used.add(dig)
            # Обновляем значение, с которого начинается результирующий список
            begin = min(dig, begin)

    # Значение, которым должен заканчиваться результирующий список
    end: int = (len(result) + begin) - 1
    # Генератор чисел, которые нужно подставить,
    # чтобы получить возрастающий список.
    missing_digits: Generator[int, None, None] = (
        n for n in range(begin, end + 1) if n not in elements_used
    )
    # Проверяем каждое значение списка: если значение не входит
    # в результирующий диапазон или продублировано, заменяем его
    # на число из генератора для подстановки
    for i, val in enumerate(result):
        if val is None or val > end:
            result[i] = next(missing_digits)
            positions.append(i)

    return (len(positions), positions, result)


# -------------------------------------------------------------------------------------------------
def get_word_palindrome(chars: Iterable[str], *, with_separator: bool = True) -> str:
    """Из заданного набора символов сформировать палиндром.

    Args:
        chars (Iterable[str]): Список символов.
        with_separator (bool): Добавлять символ-разделитель. Default: True

    Returns:
        (str): Палиндром. Если сформировать палиндром не удалось, возвращается пустая строка.
    """
    result: str = ""
    # Массив для аккумулирования кандидатов символов-разделителей между половинами палиндрома
    separator_candidate = array.array("u")

    # Внутренняя функция генератор для формирования символов, входящих в палиндром
    # В параметре передаем итератор на строку, т.к. не собираемся менять строку и копия не нужна
    def _get_palindrome_chars(chrs: Iterator[str]) -> Generator[str, None, None]:
        # Подсчитываем количество символов в заданном наборе и запускаем цикл их перебора
        for _char, _count in Counter(chrs).items():
            # Если количество символа нечетное, то это потенциальный символ-разделитель
            if with_separator and (_count & 1):
                separator_candidate.append(_char)
            # Возвращаем только символы, у которых количество пар одна и более
            if _pair_count := (_count >> 1):
                yield str(_char * _pair_count)

    # Собираем результирующий палиндром. Join работает быстрее чем конкатенация
    if half_palindrome := "".join(sorted(_get_palindrome_chars(iter(chars)))):
        result = "".join(
            (
                half_palindrome,
                # Определяем символ-разделитель как лексикографически минимальный
                min(separator_candidate, default=""),
                half_palindrome[::-1],
            )
        )
    return result


# -------------------------------------------------------------------------------------------------
TRanges: TypeAlias = list[tuple[int, int]]


def get_minmax_ranges(numbers: Iterable[TNumber]) -> dict[str, TRanges]:
    """Алгоритм поиска в заданном списке непрерывной последовательности чисел,
    сумма которых минимальна/максимальна. Заданный список может содержать как положительные,
    так и отрицательные значения, повторяющиеся и нулевые. Предварительная сортировка не требуется.

    Алгоритм за один проход по заданному списку одновременно находит минимальную и максимальную
    суммы и все возможные комбинации непрерывных диапазонов чисел, формирующие эти суммы.

    Для примера на максимальной сумме, суть алгоритма в следующем:
    Числа из списка накопительно суммируются. Если очередная накопительная сумма становится
    отрицательной, то это означает, что предыдущий диапазон чисел можно отбросить.
    Например:  [1, 2, -4, 5, 3, -1]  Первые три значения дают отрицательную сумму.
    Значит максимальная сумма возможна только со следующего значения (5 и т.д.).
    Первые три числа исключаются. На третьем шаге (-4) накопительная сумма обнуляется,
    начальный индекс сдвигается на четвертую позицию (5) и накопительная сумма
    формируется заново.

    Получение нулевой накопительной суммы, означает один из вариантов для максимальной суммы.
    Например, для списка [-1, 3, -2, 5, -6, 6] максимальная сумма 6 может быть получена из
    (3, -2, 5), (3, -2, 5, -6, 6) и (6). На значении -6 накопительная сумма становится равной нулю,
    что приводит к добавлению дополнительного начального индекса, указывающего на следующее число (6).
    Для каждого варианта в списке сохраняются начальные индексы, из которых формируются список пар
    начальных и конечных индексов. В данном примере список пар индексов: (1, 3), (1, 5), (5, 5)

    Args:
        digits (Iterable[TDigit]): Заданный список чисел.

    Returns:
        (dict[str, list[tuple[int, int]]]): Словарь, в качестве ключей содержащий минимальную и максимальную
        суммы, а в качестве значений список пар диапазонов чисел. Внимание!!! Конечный индекс закрытый и указывает
        на число, входящее в диапазон. Для итерации конечный индекс необходимо нарастить на единицу.
    """
    iter_numbers: Iterator[TNumber] = iter(numbers)
    # Из итератора получаем первый элемент для инициализации
    try:
        first_digit: TNumber = next(iter_numbers)
    except StopIteration:
        return dict()  # Если список исходных данных пуст

    class SumMode(Enum):
        MIN = "Min sum: "
        MAX = "Max sum: "

        def __str__(self):
            return self.value

    class Sum:
        """
        Внутренний класс, вычисляющий минимальную или максимальную сумму в зависимости
        от заданного режима.
        """

        __slots__ = (
            "__mode",
            "__sum",
            "__accumulated",
            "__ranges",
            "__begin_list",
        )

        def __init__(self, init_number: TNumber, mode: SumMode) -> None:
            self.__mode: SumMode = mode
            # Искомая минимальная или максимальная сумма
            self.__sum: TNumber = init_number
            self.__accumulated: TNumber = cast(TNumber, 0)  # Накопитель суммы
            # Список пар индексов begin/end диапазона чисел, составляющих сумму
            self.__ranges: TRanges = list()
            # Список начальных индексов, т.к. одна и та же сумма может быть получена из разного набора цифр
            self.__begin_list: list[int] = [0]

        def __call__(self, index: int, number: TNumber) -> None:
            self.__accumulated += number
            # Если накопленная сумма больше/меньше или сравнялась с ранее сохраненной
            if not (
                self.__accumulated < self.__sum
                if self.__mode == SumMode.MAX
                else self.__accumulated > self.__sum
            ):
                # Если накопленная сумма больше/меньше
                if (
                    self.__accumulated > self.__sum
                    if self.__mode == SumMode.MAX
                    else self.__accumulated < self.__sum
                ):
                    # Сохраняем накопленную сумму как искомую
                    self.__sum = self.__accumulated
                    # Сбрасываем список диапазонов суммы и формаруем новый
                    self.__ranges.clear()
                # Если накопленная сумма больше/меньше или равна, то формируем список пар начальных и конечных индексов
                for i in self.__begin_list:
                    self.__ranges.append((i, index))
            # Если накопленная сумма отрицательная/положительная или нулевая
            if not (
                self.__accumulated > 0
                if self.__mode == SumMode.MAX
                else self.__accumulated < 0
            ):
                # При отрицательной/положительной накопленной сумме
                if (
                    self.__accumulated < 0
                    if self.__mode == SumMode.MAX
                    else self.__accumulated > 0
                ):
                    self.__accumulated = cast(TNumber, 0)  # Обнуляем накопленную сумму
                    self.__begin_list.clear()  # Очищаем список начальных индексов
                # При отрицательной/положительной накопленной сумме формируем список начальных индексов заново.
                # При нулевой - добавляем новый начальный индекс
                self.__begin_list.append(index + 1)

        @property
        def sum(self) -> TNumber:
            return self.__sum

        @property
        def ranges(self) -> TRanges:
            return self.__ranges

        @property
        def mode(self) -> SumMode:
            return self.__mode

    # Инициализируем первым элементом списка
    min_sum = Sum(first_digit, SumMode.MIN)
    max_sum = Sum(first_digit, SumMode.MAX)

    # С помощью chain возвращаем первый элемент в итератор и запускаем цикл перебора значений
    for idx, number in enumerate(chain((first_digit,), iter_numbers)):
        min_sum(idx, number)
        max_sum(idx, number)

    # Для результирующего словаря в качестве ключей используем строковые значения,
    # т.к. минимальная и максимальная суммы могут быть равны.
    return {
        f"{min_sum.mode}{min_sum.sum}": min_sum.ranges,
        f"{max_sum.mode}{max_sum.sum}": max_sum.ranges,
    }


# -------------------------------------------------------------------------------------------------


def get_max_from_min_difference(
    groups: int, members: int, data: Iterable[int]
) -> int | None:
    """Найти заданное число групп с минимальными разницами между числами в группах и из них выбрать
      группу с максимальной разницей. Вернуть значение максимальной разницы найденной группы.

    Details:
        Формируется список пар (ключ, значение) всех возможных групп. В каждой группе числа отсортированы.
        Значения - это разница между последним и первым числом в группе. Ключи - индекс группы в исходном
        предварительно отсортированном массиве данных. Ключи дополнительно играют роль определителей
        пересечения диапазонов индексов групп, т.к. одно и то же число может содержаться в нескольких
        группах. Необходимо будет отбирать группы с непересекающимися диапазонами чисел, для чего понадобятся
        ключи и шаг непересекающихся диапазонов.

    Examples:
        Пример 1: groups=3, members=3, data=[1, 1, 1, 2, 2, 2, 2, 10, 10, 11], result=1
        Пример 2: groups=3, members=3, data=[1, 1, 1, 2, 2, 2, 2, 10, 10], result=8
        Пример 3: groups=2, members=3, data=[1, 1, 2, 2, 2, 3, 3], result=1
        Пример 4: groups=2, members=3, data=[170, 205, 225, 190, 260, 130, 225, 160], result=30

    Args:
        groups (int): Число групп
        members (int): Размер группы
        data (Sequence[int]): Массив чисел

    Returns:
        int | None: Найденная разница между числами или None, если поиск безуспешен
    """
    # Массив исходных данных обязательно должен быть отсортирован, чтобы значения в каждой
    # группе были по возрастанию
    _data: list[int] = sorted(data)
    len_data: int = len(_data)
    size_groups: int = groups * members
    # Если сформировать требуемое количество групп невозможно, досрочно выходим
    if len_data == 0 or size_groups > len_data or size_groups <= 0:
        return None
    # Вычисляем смещение, начиная с которого ищем непересекающуюся группу с результирующей разницей
    # Пример 1: 6
    # Пример 2: 6
    # Пример 3: 3
    # Пример 4: 3
    offset: int = members * (groups - 1)
    # Формируем список всех возможных групп заданного размера (members), одновременно сортируя
    # ключи по возрастанию величины разницы между числами в группе. В список отбираются только
    # те ключи, которые позволяют сформировать заданное число групп.
    # Пример 1: [(0, 0), (1, 1), (7, 1), (6, 8)]
    # Пример 2: [(0, 0), (6, 8)]
    # Пример 3: [(0, 1), (1, 1), (3, 1), (4, 1)]
    # Пример 4: [(4, 20), (1, 30), (2, 35), (3, 35), (5, 35), (0, 40)]
    diff_groups: list[tuple[int, int]] = sorted(
        (
            (a, _data[b] - _data[a])
            for a, b in enumerate(range(members - 1, len_data))
            if a >= offset or a <= (len_data - size_groups)
        ),
        key=lambda item: item[1],
    )
    # Отсортированный список больше не нужен
    del _data
    # Достаточно отобрать только первый ключ, который должен иметь пару с заданным смещением
    # Пример 1: 0
    # Пример 2: 0
    # Пример 3: 0
    # Пример 4: 4
    for start_key, _ in diff_groups:
        # Ищем ближайшую пару для стартового ключа с заданным смещением, которое зависит от
        # количества и размера групп. Найденная пара является той группой, разница чисел в
        # которой и есть искомый результат. Если найти группу невозможно, возвращаем None.
        # Пример 1: 7
        # Пример 2: 6
        # Пример 3: 3
        # Пример 4: 1
        return next(
            (value for key, value in diff_groups if abs(key - start_key) >= offset),
            None,
        )


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
        f" get_min_permutations([10, 31, 15, 22, 14, 17, 16], [16, 22, 14, 10, 31, 15, 17])) -> {get_min_permutations([10, 31, 15, 22, 14, 17, 16], [16, 22, 14, 10, 31, 15, 17])}"
    )

    print("\n- Подсчет количества нахождений заданного элемента в списке.")
    print(
        f" count_items([0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 'a', 'a', 'a'], '0') -> {count_items([0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 'a', 'a', 'a'], '0')}"
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
    print(f" get_word_palindrom('bbaadcbb') -> {get_word_palindrome('bbaadcbb')}")

    print(
        "\n- Найти минимальную и максимальную суммы и диапазоны цифр их составляющих."
    )
    print(
        f" get_minmax_ranges([-1, 3, -2, 5, -6, 6, -6]) -> {get_minmax_ranges([-1, 3, -2, 5, -6, 6, -6])}"
    )

    print(
        "\n- Найти заданное число групп с минимальными разницами между числами в группах и выбрать группу с максимальной разницей."
    )
    print(
        f" get_max_from_min_difference(3, 3, [1, 1, 1, 2, 2, 2, 2, 10, 10]) -> {get_max_from_min_difference(3, 3, [1, 1, 1, 2, 2, 2, 2, 10, 10])}"
    )


if __name__ == "__main__":
    main()
