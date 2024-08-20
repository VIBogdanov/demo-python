from collections import defaultdict, deque
from collections.abc import Collection, Iterable, Iterator
from enum import Enum
from functools import reduce
from itertools import accumulate
from typing import Any, NamedTuple, TypeAlias, TypeVar

from demo import get_positive_int, is_int, type_checking

T = TypeVar("T")
TNumber: TypeAlias = int | float | str
TInt: TypeAlias = int | str


# ------------------------------------------------------------------------------
def find_intervals(
    elements: Iterable[int],
    target: int,
) -> list[tuple[int, int]]:
    """
    Поиск в списке из чисел последовательного непрерывного интервала(-ов) чисел,
    сумма которых равна искомому значению.
    Суть алгоритма выражена формулой: Sum1 - Sum2 = Target >> Sum1 - Target = Sum2
      - Вычислить все суммы от начала до текущей позиции.
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
        >>> find_intervals([1, -3, 4, 5], 9) -> [(2, 3)]

        >>> find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0) -> [(0, 1), (4, 6), (8, 9), (9, 10)]
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
    number: TNumber,
    *,
    previous: bool = True,
) -> int | None:
    """
    Функция поиска ближайшего целого числа, которое меньше или больше заданного
    и состоит из тех же цифр.

    Args:
        number (int | float | str): Целое число, относительно которого осуществляется
        поиск. Допускается строковое представление числа, положительные или
        отрицательные значения.

        previous (bool, optional): Направление поиска: ближайшее меньшее или
        большее. По-умолчанию True - ближайшее меньшее.

    Returns:
        (int | None): Если поиск безуспешен, возвращается значение None.

    Example:
        >>> find_nearest_number(273145) -> 271543

        >>> find_nearest_number(273145, previous=False) -> 273154

        >>> find_nearest_number(-273145) -> -273154

    """
    # если входное значение невозможно представить как целое число, возвращаем None
    try:
        input_number: int = int(number)
    except (ValueError, TypeError):
        return None
    else:
        # по-умолчанию, в случае безуспешного поиска, возвращаем None
        result: int | None = None
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
            if (
                res := _do_find_nearest(digits_list, i, previous=is_previous)
            ) is not None:
                results_list.append(res)

        if results_list:
            # Если список результирующих чисел не пуст, находим наибольшее или наименьшее число
            # в зависимости от направления поиска и восстанавливаем знак числа.
            result = (
                max(results_list) if is_previous else min(results_list)
            ) * sign_number

        return result


# -------------------------------------------------------------------------------
def _do_find_nearest(
    digits_list: Iterable[int],
    current_index: int,
    previous: bool = True,
) -> int | None:
    """
    Вспомогательная подпрограмма для функции find_nearest_number. Просматривает
    цифры левее текущей цифры исходного числа с целью поиска большего или
    меньшего значения в зависимости от направления поиска. В случае успешного поиска,
    выполняет перестановку цифр и сортирует правую часть числа по возрастанию или
    убыванию в зависимости от направления поиска.

    Args:
        digits_list (Iterable[int]): Массив цифр исходного числа

        current_index (int): Текущая позиция цифры, относительно которой выполняется перестановка

        previous (bool): Направление поиска: ближайшее большее или меньшее. True - меньшее, False - большее

    Returns:
        (int | None): Возвращает найденное целое число или None в случае
        безуспешного поиска

    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    _digits_list: list[int] = list(filter(is_int, digits_list))

    i: int = current_index  # текущая позиция исходного числа, относительно которой ведется поиск
    for k in range(i - 1, -1, -1):  # просматриваем все цифры левее текущей позиции
        # сравниваем с текущей позицией, учитывая направление поиска
        if (
            (_digits_list[k] > _digits_list[i])
            if previous
            else (_digits_list[i] > _digits_list[k])
        ):
            # в случае успешного сравнения, переставляем местами найденную цифру с текущей
            _digits_list[k], _digits_list[i] = _digits_list[i], _digits_list[k]
            # если первая цифра полученного числа после перестановки не равна 0,
            # выполняем сортировку правой части числа
            if _digits_list[0] > 0:
                k += 1  # правая часть числа начинается со сдвигом от найденной позиции
                # сортируем правую часть числа (по возрастанию или по убыванию) с учетом направления поиска
                _digits_list[k::] = sorted(iter(_digits_list[k::]), reverse=previous)
                # собираем из массива цифр результирующее число
                return reduce(
                    lambda dig_prev, dig_next: 10 * dig_prev + dig_next, _digits_list
                )
    return None


# ------------------------------------------------------------------------------------
def find_item_by_binary(
    elements: Collection[Any],
    target: Any,
) -> int | None:
    """
    Функция поиска заданного значения в одномерном массиве. В качестве алгоритма поиска
    используется, так называемый, алгоритм бинарного поиска. Суть алгоритма: на каждой
    итерации сравнивать срединный элемент массива с искомым значением. Далее выяснить
    в какой из половинок массива находится искомое значение и выбрать эту половину
    для дальнейшего деления, пока не будет найдено совпадение.

    Внимание!!! Входной массив данных обязательно должен быть отсортирован. Функция учитывает
    направление сортировки - по возрастанию или убыванию.

    Args:
        elements (Sequence[Any]): Массив данных для поиска
        target (Any): Значение, которое необходимо найти

    Returns:
        int | None: Функция возвращает индекс элемента в массиве, который равен искомому значению.
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
        # Делим текущий остаток массива пополам
        i_middle: int = (i_first + i_last) >> 1
        try:
            # Сравниваем срединный элемент с искомым значением
            # Смещаем начальный или конечный индексы в зависимости
            # от результата сравнения и от направления сортировки
            if elements[i_middle] == target:
                i_target = i_middle
            elif elements[i_middle] > target:
                i_first, i_last = (
                    (i_first, i_middle - 1) if is_forward else (i_middle + 1, i_last)
                )
            else:
                i_first, i_last = (
                    (i_middle + 1, i_last) if is_forward else (i_first, i_middle - 1)
                )
        # Обрабатываем исключение в случае невозможности сравнить искомое значение с элементом массива
        except (ValueError, TypeError):
            return None

    return i_target


# ---------------------------------------------find_item_by_interpolation---------------------------------------------
def find_item_by_interpolation(
    elements: Collection[int | float],
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
    sort_order: int = 1 if elements[-1] >= elements[0] else -1
    # Стартуем с первого и последнего индекса массива одновременно
    i_first: int = 0
    i_end: int = len(elements) - 1
    i_target: int | None = None  # Возвращаемый индекс найденого значения

    while i_first <= i_end and i_target is None:
        # Если искомый элемент вне проверяемого диапазона, выходим
        # Эта проверка необходима, чтобы избежать зацикливания при неотсортированном исходном списке
        if ((sort_order * target) < (sort_order * elements[i_first])) or (
            (sort_order * target) > (sort_order * elements[i_end])
        ):
            return i_target

        try:
            # Пытаемся вычислить положение искомого элемента в списке. При этом не важно направление сортировки.
            # Возможно деление на ноль, которое перехватывается в блоке except.
            i_current = i_first + int(
                (
                    ((i_end - i_first) / (elements[i_end] - elements[i_first]))
                    * (target - elements[i_first])
                )
            )

            match ...:  # Сравниваем вычисленный элемент с искомым значением
                # Если искомое значение найдено, прекращаем дальнейший поиск и возвращаем найденный индекс
                case _ if elements[i_current] == target:
                    i_target = i_current
                # В двух других случаях смещаем начальный или конечный индексы в зависимости от
                # результата сравнения текущего элемента с искомым значением и от направления сортировки
                case _ if elements[i_current] > target:
                    i_first, i_end = (
                        (i_first, i_current - 1)
                        if sort_order == 1
                        else (i_current + 1, i_end)
                    )
                case _ if elements[i_current] < target:
                    i_first, i_end = (
                        (i_current + 1, i_end)
                        if sort_order == 1
                        else (i_first, i_current - 1)
                    )
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
    Функция сортировки методом пузырька. В отличии от классического метода, функция за каждую итерацию
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
    # Индексы первого и последнего элементов
    i_start: int = 0
    i_end: int = len(_elements) - 1
    # Флаг, исключающий "пустые" циклы, когда список достигает состояния "отсортирован" на одной из итераций
    is_swapped = False

    while i_start < i_end:
        for i_current in range(i_start, i_end, 1):
            # Если текущий элемент больше следующего, то переставляем их местами. Это потенциальный максимум.
            if (
                (_elements[i_current + 1] > _elements[i_current])
                if revers
                else (_elements[i_current] > _elements[i_current + 1])
            ):
                _elements[i_current], _elements[i_current + 1] = (
                    _elements[i_current + 1],
                    _elements[i_current],
                )
                # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
                if i_current > i_start and (
                    (_elements[i_start] < _elements[i_current])
                    if revers
                    else (_elements[i_current] < _elements[i_start])
                ):
                    _elements[i_start], _elements[i_current] = (
                        _elements[i_current],
                        _elements[i_start],
                    )
                # Список пока не отсортирован, т.к. потребовались перестановки
                is_swapped = True

        # После каждой итерации по элементам списка, сокращаем длину проверяемого диапазона на 2,
        # т.к. на предыдущей итерации найдены одновременно минимум и максимум
        if is_swapped:
            i_start += 1
            i_end -= 1
            is_swapped = False
        else:
            # Если за итерацию перестановок не было, то список уже отсортирован. Выходим из цикла
            i_start, i_end = 0, 0

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
    # try:
    #    _elements: list = list(elements)
    # except (ValueError, TypeError):
    #    return []

    if len(elements) > 1:
        # Делим исходный список пополам.
        i_middle: int = len(elements) // 2
        # Рекурсивно вызываем функцию до тех пор,
        # пока исходный список не будет разложен поэлементно.
        left_list: list = sort_by_merge(elements[:i_middle], revers=revers)
        right_list: list = sort_by_merge(elements[i_middle:], revers=revers)
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
                elements[i_result] = left_list[i_left]
                i_left += 1
            else:
                elements[i_result] = right_list[i_right]
                i_right += 1
            i_result += 1
        # Добавляем в результирующий список "хвосты", оставшиеся от половинок.
        match (i_left < len(left_list), i_right < len(right_list)):
            case (True, False):
                elements[i_result:] = left_list[i_left:]
            case (False, True):
                elements[i_result:] = right_list[i_right:]

    return elements


# --------------------------------------------------------------------------------------------
def sort_by_merge2(elements: Iterable[T], *, revers: bool = False) -> list[T]:
    """
    Усовершенствованная версия функции сортировки методом слияния (см. sort_by_merge). В отличии
    от оригинальной версии не использует рекурсивные вызовы и не создает каскад списков.
    Вместо этого создается список индексов для диапазонов сортировки, по которым происходит отбор
    значений из списка источника и сортировка по месту.

    Args:
        elements (Iterable[T]): Список данных для сортировки.

        revers (bool, optional): Если задано True, то сортировка по убыванию. Defaults to False.

    Returns:
        list[T]: Результирующий отсортированный список.
    """

    # Вспомогательный именованный кортеж
    class MergeRanges(NamedTuple):
        first_index: int
        middle_index: int
        last_index: int

    # Создаем копию списка, на котором будем производить сортировку. Он же будет результирующим.
    try:
        _elements: list[T] = list(elements)
    except (ValueError, TypeError):
        return []

    if (_ln := len(_elements)) > 1:
        # Очереди для создания списка индексов.
        query_buff: deque = deque()
        query_work: deque = deque()
        # Инициализируем буферную очередь исходным списком, деленным пополам
        query_buff.append(MergeRanges(0, (_ln // 2), _ln))
        # Далее делим пополам обе половины до тех пор, пока в каждой половине не останется по два элемента
        while query_buff:
            i_first, i_middle, i_last = query_buff.popleft()
            # Делим пополам левую часть
            if (_md := (i_middle - i_first) // 2) > 0:
                query_buff.append(MergeRanges(i_first, (i_first + _md), i_middle))
            # Делим пополам правую часть
            if (_md := (i_last - i_middle) // 2) > 0:
                query_buff.append(MergeRanges(i_middle, (i_middle + _md), i_last))
            # Результирующая очередь будет содержать индексы диапазонов для каждой из половин
            query_work.append(MergeRanges(i_first, i_middle, i_last))
        # Сортируем все полученные половины и собираем из них результирующий отсортированный список
        while query_work:
            # Выбираем из очереди диапазоны начиная с меньших
            i_first, i_middle, i_last = query_work.pop()
            i_current: int = i_first
            # Формируем список с данными для левой половины. Правую используем из самого сортируемого списка.
            left_list: tuple = tuple(_elements[i_first:i_middle])
            # Инициализируем индексы. Левый по отдельному списку. Правый по сортируемому.
            i_left: int = 0
            i_right: int = i_middle
            # Поэлементно сравниваем половины и формируем результирующий список
            while i_left < len(left_list) and i_right < i_last:
                if (
                    (left_list[i_left] > _elements[i_right])
                    if revers  # Учитываем порядок сортировки
                    else (left_list[i_left] < _elements[i_right])
                ):
                    _elements[i_current] = left_list[i_left]
                    i_left += 1
                else:
                    _elements[i_current] = _elements[i_right]
                    i_right += 1
                i_current += 1
            # Добавляем в результирующий список "хвост" от левой половины. Правая уже в списке.
            if i_left < len(left_list):
                _elements[i_current:i_last] = left_list[i_left:]

    return _elements


# --------------------------------------------------------------------------------------------
class SortMethod(str, Enum):
    SHELL = "Shell"
    HIBBARD = "Hibbard"
    SEDGEWICK = "Sedgewick"
    KNUTH = "Knuth"
    FIBONACCI = "Fibonacci"


class GetRangesSort(Iterable):
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

    __slots__ = "__calc_res"

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
        # Итерируем в обратном порядке от большего к меньшему
        return iter(self.__calc_res[::-1])

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

    def __get_fibonacci_gen(self, ln: int) -> Iterator[int]:
        """
        Формирует отличную от классической последовательность:
        вместо [0,1,1,2,3,5...] получаем [1,2,3,5...]
        Ноль и дублирование первых двух единиц не требуется.
        """
        prev: int = 1
        curr: int = 1
        while curr <= ln:
            yield curr
            curr, prev = (prev + curr), curr


# --------------------------------------------------------------------------------------------
def sort_by_shell(
    elements: Iterable[T],
    *,
    revers: bool = False,
    method: SortMethod = SortMethod.SHELL,
) -> list[T]:
    """
    Функция сортировки методом Shell. Кроме классического метода формирования
    диапазона чисел для перестановки, возможно использовать следующие методы:
    - Hibbard
    - Sedgewick
    - Knuth
    - Fibonacci

    Реализована двунаправленная сортировка.

    Args:
        elements (Iterable): Список данных для сортировки

        revers (bool, optional): Если задано True, то сортировка по убыванию. Defaults to False.

        method (SortMethod, optional): Метод формирования диапазона: Shell, Hibbard, Sedgewick, Knuth,
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
        for range_item in GetRangesSort(len(_elements), method):
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
        # Стартуем с диапазона равного длине списка данных, кроме последнего элемента.
        i_start: int = 0
        i_end: int = len(_elements) - 1
        # Потенциальные минимум и максимум в начале и конце диапазона
        i_min: int = i_start
        i_max: int = i_end

        # Флаг, исключающий "пустые" циклы, когда список достигает состояния "отсортирован" на одной из итераций
        is_swapped = False

        def _compare(revers: bool, prev, next) -> bool:
            return prev < next if revers else next < prev

        # Перебираем диапазоны, сокращая длину каждого следующего диапазона на 2
        while i_start < i_end:
            # Т.к. до последнего элемента не доходим, необходимо перед итерацией
            # сравнить последний элемент с первым. Возможно последний элемент
            # потенциальный минимум текущего диапазона
            if _compare(revers, _elements[i_start], _elements[i_end]):
                # Меняем местами первый и последний элементы текущего диапазона
                _elements[i_start], _elements[i_end] = (
                    _elements[i_end],
                    _elements[i_start],
                )

            for i_current in range(i_start, i_end, 1):
                _current_element = _elements[i_current]
                # Если текущий элемент больше последнего в диапазоне, то это потенциальный максимум
                # для текущего диапазона.
                if _compare(revers, _current_element, _elements[i_max]):
                    i_max = i_current
                    is_swapped = True
                # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
                elif _compare(revers, _elements[i_min], _current_element):
                    i_min = i_current
                    is_swapped = True
                # Выясняем требуется ли перестановка на следующей итерации
                elif not is_swapped and _compare(
                    revers, _current_element, _elements[i_current + 1]
                ):
                    is_swapped = True

            # После каждой итерации по элементам списка, сокращаем длину проверяемого диапазона на 2,
            # т.к. на предыдущей итерации найдены одновременно минимум и максимум
            if is_swapped:
                # Если найдены потенциальные минимум и/или максимум, выполняем перестановки элементов
                # с начальным и/или конечным элементом текущего диапазона.
                if i_max != i_end:
                    _elements[i_end], _elements[i_max] = (
                        _elements[i_max],
                        _elements[i_end],
                    )
                if i_min != i_start:
                    _elements[i_start], _elements[i_min] = (
                        _elements[i_min],
                        _elements[i_start],
                    )
                i_start += 1
                i_end -= 1
                i_min = i_start
                i_max = i_end
                is_swapped = False
            else:
                # Если за итерацию перестановок не потребовалось, то список уже отсортирован. Выходим из цикла
                i_start, i_end = 0, 0

    return _elements


# -------------------------------------------------------------------------------------------------
TNumber = TypeVar("TNumber", int, str, bytes)


@type_checking(TNumber.__constraints__, TNumber.__constraints__)
def get_common_divisor(number_a: TNumber, number_b: TNumber) -> int | None:
    """
    Алгоритм нахождения наибольшего общего делителя двух целых чисел без перебора.
    Используется метод Евклида. Например, для чисел 20 и 12:
    - 20 % 12 = 8
    - 12 % 8 = 4
    - 8 % 4 = 0

    Искомый делитель равен 4

    Порядок следования входных значений не важен. Допускаются как положительные, так и отрицательные
    входные значения чисел в различных комбинациях. Возможны комбинации нулевых входных значений.

    Args:
        number_a (int): Первое число

        number_b (int): Второе число

    Returns:
        int: Наибольший общий делитель. Если делитель равен 0, возвращает None.
    """
    # На всякий случай
    number_a = get_positive_int(number_a)
    number_b = get_positive_int(number_b)
    # Определяем делимое и делитель. Делимое - большее число. Делитель - меньшее.
    divisible, divisor = (
        (number_a, number_b)
        if max(number_a, number_b) == number_a
        else (number_b, number_a)
    )

    # Ищем общий делитель как остаток от деления, при котором на следующей итерации остаток от деления равен 0.
    while divisor:
        divisible, divisor = divisor, divisible % divisor

    return divisible if bool(divisible) else None


# --------------------------------------------------------------------------------------------
TDigit = TypeVar("TDigit", int, str, bytes)


def find_pairs_sum(
    digits: Iterable[TDigit],
    target: TDigit,
) -> list[tuple[int, int]]:
    """В заданном наборе чисел найти неповторяющиеся пары целых чисел, сумма которых равна целевому значению.

    Details:
    Допускаются отрицательные, нулевые и повторяющиеся числа. Предварительная сортировка не требуется.

    Args:
        digits (Iterable): Набор чисел
        target: Целевое значение

    Returns:
        (list[tuple[int, int]]): Список пар
    """
    result_list: list[tuple[int, int]] = list()
    # Удаляем дубли и сортируем входной набор чисел. Сортировка обязательна!!!
    digits_list: list[int] = sorted(set(int(i) for i in filter(is_int, digits)))
    # Приводим входные параметры к единому типу int
    try:
        _target: int = int(target)
    except (ValueError, TypeError):
        return result_list

    # Задаем индексы первого и последнего числа списка
    i_begin = 0
    i_end: int = len(digits_list) - 1
    # Запускаем цикл встречного перебора
    while i_begin <= i_end:
        # Если сумма больше целевого значения, сдвигаем конечный индекс к началу списка
        if (pair_sum := digits_list[i_begin] + digits_list[i_end]) > _target:
            i_end -= 1
        # Найдена искомая сумма. Сохраняем пару слагаемых и сдвигаем конечный индекс к началу
        elif pair_sum == _target:
            result_list.append((digits_list[i_begin], digits_list[i_end]))
            i_end -= 1
        # Если сумма меньше, наращиваем начальный индекс
        else:
            i_begin += 1

    return result_list


# --------------------------------------------------------------------------------------------
def main():
    print(
        "\n- Функция нахождения наибольшего общего делителя двух целых чисел без перебора методом Евклида."
    )
    print(f" get_common_divisor(20, 12) -> {get_common_divisor(20, 12)}")

    print("\n- Поиск элемента в массиве данных при помощи бинарного алгоритма.")
    print(
        f" find_item_by_binary([-20, 30, 40, 50], 30) -> {find_item_by_binary([-20, 30, 40, 50], 30)}"
    )

    print("\n- Поиск элемента в массиве данных при помощи алгоритма интерполяции.")
    print(
        f" find_item_by_interpolation([-1, -2, 3, 4, 5], 4) -> {find_item_by_interpolation([-1, -2, 3, 4, 5], 4)}"
    )

    print(
        "\n- Поиск в списке из чисел последовательного непрерывного интервала(-ов) чисел, сумма которых равна искомому значению."
    )
    print(" find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0) -> ", end="")
    for res in find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0):
        print(tuple(res), end=" ")
    print("")

    print(
        "\n- Поиск ближайшего целого числа, которое меньше или больше заданного и состоит из тех же цифр."
    )
    print(f" find_nearest_number(273145) -> {find_nearest_number(273145)}")

    print("\n- Сортировки методом пузырька.")
    print(
        f" sort_by_bubble([2, 7, 3, 1, 4, 5]) -> {sort_by_bubble([2, 7, 3, 1, 4, 5])}"
    )

    print("\n- Сортировки методом слияния.")
    print(f" sort_by_merge([2, 7, 3, 1, 4, 5]) -> {sort_by_merge([2, 7, 3, 1, 4, 5])}")

    print("\n- Усовершенствованная версия сортировки методом слияния.")
    print(
        f" sort_by_merge2([2, 7, 3, 1, 4, 5]) -> {sort_by_merge2([2, 7, 3, 1, 4, 5])}"
    )

    print("\n- Сортировки методом Shell.")
    print(
        f" sort_by_shell([2, 7, 3, 1, 4, 5]) -> {sort_by_shell([2, 7, 3, 1, 4, 5], method = 'Shell')}"
    )

    print("\n- Сортировки методом выбора.")
    print(
        f" sort_by_selection([2, 7, 3, 1, 4, 5]) -> {sort_by_selection([2, 7, 3, 1, 4, 5])}"
    )

    print("\n- Поиск неповторяющихся пар чисел, сумма которых равна целевому значению.")
    print(
        f" find_pairs_sum([3, 1, 2, 3, 0, 2, -1, 5, 4, 7, 6], 5) -> {find_pairs_sum([3, 1, 2, 3, 0, 2, -1, 5, 4, 7, 6], 5)}"
    )


if __name__ == "__main__":
    main()
