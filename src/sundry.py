from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from functools import reduce
from itertools import accumulate
from typing import Any, TypeAlias, TypeVar

T = TypeVar("T")
NumberValue: TypeAlias = int | float | str


# ------------------------------------------------------------------------------
def find_intervals(
    elements: Iterable[int],
    *,
    target: int = 0,
) -> list[tuple[int, int]]:
    """
    Поиск в списке из чисел последовательного непрерывного интервала(-ов) чисел,
    сумма которых равна искомому значению.
    Суть алгоритма выражена формулой: Sum1 - Sum2 = Target >> Sum1 - Target = Sum2
      - Вычислить все суимы от начала до текущей позиции.
      - Для каждой суммы вычислить Sum - Target
      - Найти полученное значение в списке сумм
      - Если пара найдена, извлечь индексы и составить диапазон

    Args:
        elements (Iterable[int]): Список неупорядоченных целых чисел, включая отрицательные значения.

        target (int): Искомое целое число, для которого ищется сумма элементов списка.

    Returns:
        list[tuple[int, int]]: Результирующий список диапазонов элементов. Диапазоны задаются в виде
        кортежей пары целых чисел, обозначающих индексы элементов списка, включая начальный и
        конечный индексы включительно. Если ни один диапазон не найден, возвращается пустой список.

    Example:
        >>> find_intervals([1, -3, 4, 5], 9)
        [(2, 3)]

        >>> find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0)
        [(0, 1), (4, 6), (8, 9), (9, 10)]
    """
    try:
        _target: int = int(target)
    except (ValueError, TypeError):
        return []
    else:
        sum_dict: defaultdict = defaultdict(list[int])
        result_list: list[tuple[int, int]] = list()

        # Суммируем элементы списка по нарастающей и сохраняем в словаре
        # список индексов для каждой суммы. В качестве ключа - сама сумма.
        for id_to, sum_accum in enumerate(accumulate(elements)):
            # Если на очередной итерации полученная сумма равна искомому значению,
            # заносим диапазон от 0 до текущей позиции в результирующий список.
            if sum_accum == _target:
                result_list.append((0, id_to))
            # Ищем пару из уже вычисленных ранее сумм для значения (Sum - Target).
            # Если пара найдена, извлекаем индексы и формируем результирующие диапазоны.
            for id_from in sum_dict.get((sum_accum - _target), []):
                result_list.append((id_from + 1, id_to))
            # Сохраняем очередную сумму и ее индекс в словаре, где ключ - сама сумма.
            sum_dict[sum_accum].append(id_to)

        return result_list


# -------------------------------------------------------------------------------
def find_nearest_number(
    number: NumberValue,
    *,
    previous: bool = True,
) -> int | None:
    """
    Функция поиска ближайшего целого числа, которое меньше или больше заданного
    и состоит из тех же цифр.

    Args:
        number (int | float | str): Целое число, относительнго которого осуществляется
        поиск. Допускается строковое представление числа, положительные или
        отрицательные значения.

        previous (bool, optional): Направление поиска: ближайшее меньшее или
        большее. По-умолчанию True - ближайшее меньшее.

    Returns:
        (int | None): Если поиск безуспешен, возвращается значение None.

    Example:
        >>> find_nearest_number(273145)
        271543

        >>> find_nearest_number(273145, previous=False)
        273154

        >>> find_nearest_number(-273145)
        -273154

    """
    # если входное значение невозможно представить как целое число, возвращаем None
    try:
        input_number: int = int(number)
    except (ValueError, TypeError):
        return None
    else:
        result: int | None = None  # по-умолчанию, в случае безуспешного поиска, возвращаем None
        is_previous: bool = previous
        sign_number: int = 1

        if input_number < 0:  # если входное число отрицательное
            sign_number = -1  # сохраняем знак числа
            input_number *= -1  # переводим входное число на положительное значение
            is_previous = not is_previous  # меняем направление поиска

        # формируем массив цифр из входного числа
        digits_list: tuple[int, ...] = tuple(map(int, str(input_number)))
        results_list: list[int] = list()  # список для накопления результатов поиска
        # цикл перебора цифр входного числа справа на лево (с хвоста к голове) кроме первой цифры
        for i in range(len(digits_list) - 1, 0, -1):
            # вызываем подпрограмму поиска большего или меньшего числа в зависимости от направления поиска
            if (res := _do_find_nearest(digits_list, i, previous=is_previous)) is not None:
                results_list.append(res)

        if results_list:
            # Если список результирующих чисел не пуст, находим наибольшее или наименьшее число
            # в зависимости от направления поиска и восстанавливаем знак числа.
            result = (max(results_list) if is_previous else min(results_list)) * sign_number

        return result


def _do_find_nearest(
    digits_list: Iterable[int],
    current_index: int,
    previous: bool = True,
) -> int | None:
    """
    Вспомогательная подпрограмма для функции find_nearest_number. Просматривает
    цифры левее текущей позиции исходного числа с целью поиска большего или
    меньшего значения в зависимости от направления поиска. В случае успешного поиска,
    выполняет перестановку цифр и сортирует правую часть числа по возрастанию или
    убыванию в зависимости от направления поиска.

    Args:
        digits_list (Iterable[int]): Массив цифр исходного числа

        current_index (int): Текущая позиция исходного числа

        previous (bool): Направление поиска: ближайшее большее или меньшее. True - меньшее, False - большее

    Returns:
        (int | None): Возвращает найденное целое число или None в случае
        безуспешного поиска

    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        dig_list: list[int] = list(digits_list)
    except (ValueError, TypeError):
        return None

    i: int = current_index  # текущая позиция исходного числа, относительно которой ведется поиск
    for k in range(i - 1, -1, -1):  # просматриваем все цифры левее текущей позиции
        # сравниваем с текущей позицией, учитывая направление поиска
        if (dig_list[k] > dig_list[i]) if previous else (dig_list[i] > dig_list[k]):
            # в случае успешного сравнения, переставляем местами найденную цифру с текущей
            dig_list[k], dig_list[i] = dig_list[i], dig_list[k]
            # если первая цифра полученного числа после перестановки не равна 0,
            # выполняем сортировку правой части числа
            if dig_list[0] > 0:
                k += 1  # правая часть числа начинается со сдвигом от найденной позиции
                # сортируем правую часть числа (по возрвстанию или по убыванию) с учетом направления поиска
                dig_list[k::] = sorted(iter(dig_list[k::]), reverse=previous)
                # собираем из массива цифр результирующее число
                return reduce(lambda dig_prev, dig_next: 10 * dig_prev + dig_next, dig_list)
    return None


# ------------------------------------------------------------------------------------
def find_item_by_binary(
    elements: Sequence,
    target: Any,
) -> int | None:
    """
    Функция поиска заданного значения в одномерном массиве. В качестве алгоритма поиска используется,
    так называемый, алгоритм бинарного поиска. Суть алгоритма: на каждой итерации сравнивать срединный
    элемент массива с искомым значением. Далее выяснить в какой из половинок массива находится искомое
    значение и выбрать эту половину для дальнейшего деления, пока не будет найдено совпадение.

    Внимание!!! Входной массив данных обязательно должен быть отсортирован. Функция учитывает
    направление сортировки - по возрастанию или убыванию.

    Args:
        elements (Sequence): Массив данных для поиска
        target (Any): Значение, которое необходимо найти

    Returns:
        int | None: Функция dозвращает индекс элемента в массиве, который равен искомому значению.
        В случае неудачного поиска, возвращается None.
    """
    # Исключаем пустые и односимвольные списки
    match len(elements):
        case 0:
            return None
        case 1:
            try:
                return 0 if elements[0] == target else None
            except (ValueError, TypeError):
                return None
    # Определяем порядок сортировки исходного массива
    is_forward: bool = bool(elements[-1] >= elements[0])
    # Стартуем с первого и последнего индекса массива одновременно
    i_first: int = 0
    i_last: int = len(elements) - 1

    i_target: int | None = None  # Возвращаемый индекс найденого значения

    while i_first <= i_last and i_target is None:
        i_current: int = (i_first + i_last) // 2  # Делим текущий остаток массива пополам
        try:
            match (elements[i_current], target):  # Сравниваем срединный элемент с искомым значением
                # Смещаем начальный или конечный индексы в зависимости от результата сравнения
                # текущего элемента с искомым значением и от направления сортировки
                case (cur, trg) if cur > trg:
                    i_first, i_last = (i_first, i_current - 1) if is_forward else (i_current + 1, i_last)
                case (cur, trg) if cur < trg:
                    i_first, i_last = (i_current + 1, i_last) if is_forward else (i_first, i_current - 1)
                case _:  # В противном случае искомое значение найдено
                    i_target = i_current
        # Обрабатываем исключение в случае невозможности сравнить искомое значение с элементом массива
        except (ValueError, TypeError):
            return None

    return i_target


# ---------------------------------------------find_item_by_interpolation---------------------------------------------
def find_item_by_interpolation(
    elements: Sequence[int | float],
    target: int | float,
) -> int | None:
    """
    Функция поиска заданного значения в одномерном числовом массиве. В качестве алгоритма поиска используется
    метод интерполяции. Алгоритм похож на бинарный поиск. Отличие в способе поиска срединного элемента.
    При интерполяции производится попытка вычислить положение искомого элемента, а не просто поделить список
    пополам.

    Внимание!!! Входной массив данных обязательно должен быть отсортирован. Функция учитывает
    направление сортировки - по возрастанию или убыванию. Т.к. в алгоритме используются арифметические
    операции, список и искомое значение должны быть числовыми. Строковые литералы не поддерживаются.

    Args:
        elements (Sequence[int | float]): Массив числовых данных для поиска
        target (int | float): Значение, которое необходимо найти

    Returns:
        int | None: Индекс элемента в массиве, который равен искомому значению.
        В случае неудачного поиска, возвращается None.
    """
    # Исключаем пустые и односимвольные списки
    match len(elements):
        case 0:
            return None
        case 1:
            try:
                return 0 if elements[0] == target else None
            except (ValueError, TypeError):
                return None
    # Определяем порядок сортировки исходного массива
    sort_order: int = 1 if elements[-1] > elements[0] else -1
    # Стартуем с первого и последнего индекса массива одновременно
    i_first: int = 0
    i_end: int = len(elements) - 1
    i_target: int | None = None  # Возвращаемый индекс найденого значения

    while i_first <= i_end and i_target is None:
        # Если искомый элемент вне проверяемого диапазона, выходим из цикла
        if ((sort_order * target) < (sort_order * elements[i_first])) or (
            (sort_order * target) > (sort_order * elements[i_end])
        ):
            return i_target

        try:
            # Пытаемся вычислить положение искомого элемента в списке. При этом не важно направление сортировки.
            # Возможно деление на ноль, которое перехватывается в блоке except.
            i_current = i_first + int(
                (((i_end - i_first) / (elements[i_end] - elements[i_first])) * (target - elements[i_first]))
            )

            match ...:  # Сравниваем срединный элемент с искомым значением
                # Если искомое значение найдено, прекращаем дальнейший поиск и возвращаем найденный индекс
                case _ if elements[i_current] == target:
                    i_target = i_current
                # В двух других случаях смещаем начальный или конечный индексы в зависимости от
                # результата сравнения текущего элемента с искомым значением и от направления сортировки
                case _ if elements[i_current] > target:
                    i_first, i_end = (i_first, i_current - 1) if sort_order == 1 else (i_current + 1, i_end)
                case _ if elements[i_current] < target:
                    i_first, i_end = (i_current + 1, i_end) if sort_order == 1 else (i_first, i_current - 1)
        # Обрабатываем исключение в случае невозможности сравнить искомое значение с элементом массива
        except (ValueError, TypeError):
            return None
        # Возможно все элементы списка одинаковые. Тогда возникает ситуация - деление на ноль
        except ZeroDivisionError:
            return i_first if elements[i_first] == target else None

    return i_target


# ----------------------------------------------------------------------------------------------------------
def sort_by_bubble(elements: Iterable[T], *, revers: bool = False) -> list[T]:
    """
    Функция сортировки по методу пузырька. В отличии от классического метода, функция за каждую итерацию
    одновременно ищет как максимальное значение, так и минимальное. На следующей итерации диапазон поиска
    сокращается не на один элемент, а на два. Кроме того, реализована сортировка как по возрастанию, так
    и по убыванию.

    Args:
        elements (Iterable): Список данных для сортировки

        revers (bool, optional): Если задано True, то сортировка по убыванию. Defaults to False.

    Returns:
        list: Возвращает отсортированный список
    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _elements: list = list(elements)
    except (ValueError, TypeError):
        return []

    i_start: int = 0
    i_end: int = len(_elements) - 1

    while i_start < i_end:
        for i_current in range(i_start, i_end, 1):
            # Если текущий элемент больше следующего, то переставляем их местами. Это потенциальный максимум.
            if (
                (_elements[i_current + 1] > _elements[i_current])
                if revers
                else (_elements[i_current] > _elements[i_current + 1])
            ):
                _elements[i_current], _elements[i_current + 1] = _elements[i_current + 1], _elements[i_current]
                # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
                if i_current > i_start and (
                    (_elements[i_start] < _elements[i_current])
                    if revers
                    else (_elements[i_current] < _elements[i_start])
                ):
                    _elements[i_start], _elements[i_current] = _elements[i_current], _elements[i_start]
        # После каждой итерации по элементам списка, сокращаем длину проверяемого диапазона на 2,
        # т.к. на предыдущей итерации найдены одновременно минимум и максимум
        i_start += 1
        i_end -= 1

    return _elements


# ------------------------------------------------------------------------------------------------
def sort_by_merge(elements: Iterable[T], *, revers: bool = False) -> list[T]:
    """
    Функция сортировки методом слияния. Поддерживается сортировка как
    по возрастанию, так и по убыванию.

    Args:
        elements (Iterable): Список данных для сортировки.
        revers (bool, optional): Если задано True, то сортировка по убыванию. Defaults to False.

    Returns:
        list: Результирующий отсортированный список.
    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _elements: list = list(elements)
    except (ValueError, TypeError):
        return []

    if len(_elements) > 1:
        # Делим исходный список пополам.
        i_middle: int = len(_elements) // 2
        # Рекурсивно вызываем функцию до тех пор,
        # пока исходный список не будет разложен поэлементно.
        left_list: list = sort_by_merge(_elements[:i_middle], revers=revers)
        right_list: list = sort_by_merge(_elements[i_middle:], revers=revers)
        # Собираем список из стека рекурсивных вызовов
        i_left: int = 0
        i_right: int = 0
        i_result: int = 0
        # Сравниваем поэлементно половинки списка и добавляем в результирующий список
        # меньший или больший элемент, в зависимости от порядка сортировки.
        while i_left < len(left_list) and i_right < len(right_list):
            if (
                (left_list[i_left] > right_list[i_right])
                if revers
                else (right_list[i_right] > left_list[i_left])
            ):
                _elements[i_result] = left_list[i_left]
                i_left += 1
            else:
                _elements[i_result] = right_list[i_right]
                i_right += 1
            i_result += 1
        # Добавляем в результирующий список "хвосты", оставшиеся от половинок.
        match (i_left < len(left_list), i_right < len(right_list)):
            case (True, False):
                _elements[i_result:] = left_list[i_left:]
            case (False, True):
                _elements[i_result:] = right_list[i_right:]

    return _elements


# --------------------------------------------------------------------------------------------
class SortMethod(str, Enum):
    SHELL = "Shell"
    HIBBARD = "Hibbard"
    SEDGEWICK = "Sedgewick"
    KNUTH = "Knuth"
    FIBONACCI = "Fibonacci"


class GetRangesSort:
    """
    Вспомогательный класс для функции sort_by_shell(). Реализует различные методы формирования
    диапазонов чисел для перестановки. Класс является итератором.
    Реализованы следующие методы:
    - Классический метод Shell
    - Hibbard
    - Sedgewick
    - Knuth
    - Fibonacci
    """

    __slots__ = ("__calc_res")

    def __init__(self, list_len: int, method: SortMethod = SortMethod.SHELL) -> None:
        self.__calc_res: list[int] = list()
        # Исходя из заданного метода, вычисляем на какие диапазоны можно разбить исходный список
        match method:
            case SortMethod.HIBBARD:
                i = 1
                while (res := (2**i - 1)) <= list_len:
                    self.__calc_res.append(res)
                    i += 1
            case SortMethod.SEDGEWICK:
                i = 0
                while (res := self.__get_sedgewick_range(i)) <= list_len:
                    self.__calc_res.append(res)
                    i += 1
            case SortMethod.KNUTH:
                i = 1
                while (res := ((3**i - 1) // 2)) <= (list_len // 3):
                    self.__calc_res.append(res)
                    i += 1
            case SortMethod.FIBONACCI:
                for res in self.__get_fibonacci_gen(list_len):
                    self.__calc_res.append(res)
            case SortMethod.SHELL | _:
                res = list_len
                while (res := (res // 2)) > 0:
                    self.__calc_res.append(res)
                else:
                    self.__calc_res.sort()

    def __iter__(self) -> Iterator[int]:  # позволяет итерировать класс
        # Возвращаемый генератор поддерживает интерфейс итератора
        # Итерируем в обратном порядке от большего к меньшему
        return (res for res in self.__calc_res[::-1])

    # позволяет применять к классу срезы и вести себя как последовательность
    def __len__(self) -> int:
        return len(self.__calc_res)

    # позволяет применять к классу срезы и вести себя как последовательность
    def __getitem__(self, index):
        match index:
            case int() | slice():
                return self.__calc_res[index]
            case _:
                try:
                    index = int(index)
                except (ValueError, TypeError):
                    return []
                else:
                    return self.__calc_res[index]

    def __get_sedgewick_range(self, i: int) -> int:
        if i % 2 == 0:
            return 9 * (2**i - 2 ** (i // 2)) + 1
        else:
            return 8 * 2**i - 6 * 2 ** ((i + 1) // 2) + 1

    def __get_fibonacci_gen(self, ln: int):
        """
        Формирует отличную от классической последовательность: вместо [1,1,2,3,5...] получаем [1,2,3,5...]
        Дублирование первых двух единиц не требуется.
        """
        prev: int = 1
        curr: int = 1
        while curr <= ln:
            yield curr
            curr, prev = (prev + curr), curr


def sort_by_shell(
    elements: Iterable[T],
    *,
    revers: bool = False,
    method: SortMethod = SortMethod.SHELL,
) -> list[T]:
    """
    Функция сортировки методом Shell. Кроме классического метода формирования
    дипазанона чисел для перестановки, возможно использовать следующие методы:
    - Hibbard
    - Sedgewick
    - Knuth
    - Fibonacci

    Реализована двунаправленная сортировка.

    Args:
        elements (Iterable): Список данных для сортировки

        revers (bool, optional): Если задано True, то сортировка по убыванию.. Defaults to False.

        method (SortMethod, optional): Мотод формирования диапазона: Shell, Hibbard, Sedgewick, Knuth,
        Fibonacci. Defaults to "Shell".

    Returns:
        list: Отсортированный список.
    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _elements: list = list(elements)
    except (ValueError, TypeError):
        return []

    if (len(_elements)) > 1:
        # Создаем экземпляр класса, который будет генерировать диапазоны выборки
        ranges_list = GetRangesSort(len(_elements), method)
        for range_item in ranges_list:
            for i_range in range(range_item, len(_elements)):
                i_current: int = i_range
                while (i_current >= range_item) and (
                    (_elements[i_current] > _elements[i_current - range_item])
                    if revers
                    else (_elements[i_current - range_item] > _elements[i_current])
                ):
                    _elements[i_current], _elements[i_current - range_item] = (
                        _elements[i_current - range_item],
                        _elements[i_current],
                    )
                    i_current -= range_item

    return _elements


# -------------------------------------------------------------------------------------------------
def sort_by_selection(elements: Iterable[T], *, revers: bool = False) -> list[T]:  # noqa: C901
    """
    Функция сортировки методом выбора. Это улучшенный вариант пузырьковой сортировки
    за счет сокращения числа перестановок элементов. Элементы переставляются не на
    каждом шаге итерации, а только лишь в конце текущей итерации. Дополнительно к
    классическому алгоритму добавлена возможность одновременного поиска максимального
    и минимального элементов текущего диапазона за одну итерацию. Реализована
    двунаправленная сортировка списка данных.

    Args:
        elements (Iterable): Список данных для сортировки.
        revers (bool, optional): Если задано True, список сортируется по убыванию. Defaults to False.

    Returns:
        list: Возвращает отсортированный список.
    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _elements: list = list(elements)
    except (ValueError, TypeError):
        return []

    if (len(_elements)) > 1:
        # Стартуем с дипазална равного длине списка данных, кроме последнего элемента.
        i_start: int = 0
        i_end: int = len(_elements) - 1
        # Потенциальные минимум и максимум в начале и конце диапазона
        i_min: int = i_start
        i_max: int = i_end

        # Перебираем диапазоны, сокращая длину каждого следующего диапазона на 2
        while i_start < i_end:
            # Т.к. до последнего элемента не доходим, необходимо перед итерацией
            # сравнить последний элемент с первым. Возмоно последний элемент
            # потенуиальный минимум текущего диапазона
            if (_elements[i_end] > _elements[i_start]) if revers else (_elements[i_start] > _elements[i_end]):
                # Меняем местами первый и последний элементы текущего диапазона
                _elements[i_start], _elements[i_end] = _elements[i_end], _elements[i_start]
            for i_current in range(i_start, i_end, 1):
                # Если текущий элемент больше последнего в диапазоне, то это потенциальный максимум
                # для текущего дипазона.
                if (_elements[i_max] > _elements[i_current]) if revers else (_elements[i_current] > _elements[i_max]):
                    i_max = i_current
                # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
                elif (i_current > i_start) and (
                    (_elements[i_min] < _elements[i_current]) if revers else (_elements[i_current] < _elements[i_min])
                ):
                    i_min = i_current
            # Если найдены потенциальные минимум и/или максимум, выполняем перестановки элементов
            # с начальным и/или конечным элементом текущего диапазона.
            if i_max != i_end:
                _elements[i_end], _elements[i_max] = _elements[i_max], _elements[i_end]
            if i_min != i_start:
                _elements[i_start], _elements[i_min] = _elements[i_min], _elements[i_start]
            # После каждой итерации по элементам списка, сокращаем длину проверяемого диапазона на 2,
            # т.к. на предыдущей итерации найдены одновременно минимум и максимум
            i_start += 1
            i_end -= 1
            i_min = i_start
            i_max = i_end

    return _elements


if __name__ == "__main__":
    lst = [4, 6, 3, 8, 1, 2, 9, 7]
    print(sort_by_shell(lst, method=SortMethod.FIBONACCI))
    pass
