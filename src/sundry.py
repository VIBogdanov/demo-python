from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import reduce
from itertools import accumulate
from typing import Any


# ------------------------------------------------------------------------------
def find_intervals(
    elements: Iterable[int],
    *,
    target: int = 0,
) -> list[tuple]:
    """
    Поиск в списке из чисел последовательного непрерывного интервала(-ов) чисел,
    сумма которых равна искомому значению.
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
        конечный индексы включительно. Если ни один диапазон не найден, возвращается пустой список.

    Example:
        >>> find_intervals([1, -3, 4, 5], 9)
        [(2, 3)]

        >>> find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0)
        [(0, 1), (4, 6), (8, 9), (9, 10)]
    """
    try:
        target = int(target)
    except (ValueError, TypeError):
        return []
    else:
        sum_dict: defaultdict = defaultdict(list[int])
        result_list: list[tuple] = list()

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


# -------------------------------------------------------------------------------
def find_nearest_number(
    number: int | float | str,
    *,
    previous: bool = True,
) -> int | None:
    """
    Функция поиска ближайшего целого числа, которое меньше или больше заданного
    и состоит из тех же цифр.

    Args:
        number (int | str): Целое число, относительнго которого осуществляется
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
        search_direction: int = 1 if previous else -1  # направление поиска: большее или меньшее
        sign_number: int = 1

        if input_number < 0:  # если входное число отрицательное
            sign_number = -1  # сохраняем знак числа
            input_number *= -1  # переводим входное число на положительное значение
            search_direction *= -1  # меняем направление поиска
        # массив цифр из входного числа
        digits_list: tuple[int, ...] = tuple(int(digit) for digit in str(input_number))
        results_list: set = set()  # список для накопления результатов поиска
        # цикл перебора цифр входного числа справа на лево (с хвоста к голове) кроме первой цифры
        for i in range(len(digits_list) - 1, 0, -1):
            # вызываем подпрограмму поиска большего или меньшего числа в зависимости от направления поиска
            results_list.add(_do_find_nearest(digits_list, i, search_direction))

        results_list.discard(None)
        if results_list:
            # если список результирующих чисел не пуст, находим наибольшее или наименьшее число
            # в зависимости от направления поиска
            result = max(results_list) if search_direction == 1 else min(results_list)

        # если искомое число найдено и входное число было отрицательным, восстанавливаем знак минус
        if result is not None and sign_number == -1:
            result *= -1

        return result


def _do_find_nearest(
    digits_list: Iterable[int],
    current_index: int,
    search_direction: int,
) -> int | None:
    """
    Вспомогательная подпрограмма для функции find_nearest_number. Просматривает
    цифры левее текущей позиции исходного числа с целью поиска большего или
    меньшего значения в зависимости от направления поиска. В случае успешного поиска,
    выполняет перестановку цифр и сортирует правую часть числа по возрастанию или
    убыванию в зависимости от направления поиска. Выделение поиска в отдельную
    подпрограмму потребовалось ради реализации мультизадачности в
    функции find_nearest_number.

    Args:
        digits_list (Sequence[int]): Массив цифр исходного числа

        current_index (int): Текущая позиция исходного числа

        search_direction (int): Направление поиска: ближайшее большее или меньшее

    Returns:
        (int | None): Возвращает найденное целое число или None в случае
        безуспешного поиска

    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _digits_list: list[int] = list(digits_list)
    except (ValueError, TypeError):
        return None

    i: int = current_index  # текущая позиция исходного числа, относительно которой ведется поиск
    for k in range(i - 1, -1, -1):  # просматриваем все цифры левее текущей позиции
        # сравниваем с текущей позицией, учитывая направление поиска
        if (search_direction * _digits_list[k]) > (search_direction * _digits_list[i]):
            # в случае успешного сравнения, переставляем местами найденную цифру с текущей
            _digits_list[k], _digits_list[i] = _digits_list[i], _digits_list[k]
            # если первая цифра полученного числа после перестановки не равна 0,
            # выполняем сортировку правой части числа
            if _digits_list[0] > 0:
                k += 1  # правая часть числа начинается со сдвигом от найденной позиции
                # сортируем правую часть числа (по возрвстанию или по убыванию) с учетом направления поиска
                _digits_list[k::] = sorted(_digits_list[k::], reverse=(search_direction == 1))
                # собираем из массива цифр результирующее число
                return reduce(lambda dig_prev, dig_next: 10 * dig_prev + dig_next, _digits_list)
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
        elements (list | tuple): Массив данных для поиска
        target (Any): Значение, которое необходимо найти

    Returns:
        int | None: Функция dозвращает индекс элемента в массиве, который равен искомому значению.
        В случае неудачного поиска, возвращается None.
    """
    _is_forward: bool = True  # По умолчанию считаем входной массив отсортированным по возрастанию
    if elements[0] > elements[-1]:
        _is_forward = False  # Иначе по убыванию
    # Стартуем с первого и последнего индекса массива
    i_first: int = 0
    i_last: int = len(elements) - 1

    i_target: int | None = None  # Возвращаемый индекс найденого значения

    while i_first <= i_last and i_target is None:
        i_current = (i_first + i_last) // 2  # Делим текущий остаток массива пополам
        try:
            match (elements[i_current], target):  # Сравниваем срединный элемент с искомым значением
                # Если искомое значение найдено, прекращаем дальнейший поиск и возвращаем найденный индекс
                case (cur, trg) if cur == trg:
                    i_target = i_current
                # В двух других случаях смещаем начальный или конечный индексы в зависимости от
                # результата сравнения текущего элемента с искомым значением и от направления сортировки
                case (cur, trg) if cur > trg:
                    i_first, i_last = (i_first, i_current - 1) if _is_forward else (i_current + 1, i_last)
                case (cur, trg) if cur < trg:
                    i_first, i_last = (i_current + 1, i_last) if _is_forward else (i_first, i_current - 1)
        # Обрабатываем исключение в случае невозможности сравнить искомое значение с элементом массива
        except (ValueError, TypeError):
            return None

    return i_target


# ----------------------------------------------------------------------------------------------------------
def sort_by_bubble(elements: Iterable, *, revers: bool = False) -> list:
    """
    Функция сортировки по методу пузырька. В отличии от классического метода, функция за каждую итерацию
    одновременно ищет как максимальное значение, так и минимальное. На следующей итерации диапазон поиска
    сокращается не на один элемент, а на два. Кроме того, реализована сортировка как по возрастанию, так
    и по убыванию.

    Args:
        elements (list): Список данных для сортировки
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
    _sort_order: int = -1 if revers else 1  # Задаем порядок сортировки

    while i_start < i_end:
        for i_current in range(i_start, i_end, 1):
            # Если текущий элемент больше следующего, то переставляем их местами. Это потенциальный максимум.
            if (_sort_order * _elements[i_current]) > (_sort_order * _elements[i_current + 1]):
                _elements[i_current], _elements[i_current + 1] = _elements[i_current + 1], _elements[i_current]
                # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
                if i_current > i_start and (_sort_order * _elements[i_current]) < (_sort_order * _elements[i_start]):
                    _elements[i_start], _elements[i_current] = _elements[i_current], _elements[i_start]
        # После каждой итерации по элементам списка, сокращаем длину проверяемого диапазона на 2,
        # т.к. на предыдущей итерации найдены одновременно минимум и максимум
        i_start += 1
        i_end -= 1

    return _elements


# ------------------------------------------------------------------------------------------------
def sort_by_merge(elements: Iterable, *, revers: bool = False) -> list:
    """
    Функция сортировки методом слияния. Поддерживается сортировка как
    по возрастанию, так и по убыванию.

    Args:
        elements (list): Список данных для сортировки.
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
        _i_middle: int = len(_elements) // 2
        # Рекурсивно вызываем функцию до тех пор,
        # пока исходный список не будет разложен поэлементно.
        _left_list: list = sort_by_merge(_elements[:_i_middle], revers=revers)
        _right_list: list = sort_by_merge(_elements[_i_middle:], revers=revers)
        # Собираем список из стека рекурсивных вызовов
        _i_left: int = 0
        _i_right: int = 0
        _i_result: int = 0
        _sort_order: int = -1 if revers else 1  # Учитываем порядок сортировки
        # Сравниваем поэлементно половинки списка и добавляем в результирующий список
        # меньший или больший элемент, в зависимости от порядка сортировки.
        while _i_left < len(_left_list) and _i_right < len(_right_list):
            if (_sort_order * _left_list[_i_left]) < (_sort_order * _right_list[_i_right]):
                _elements[_i_result] = _left_list[_i_left]
                _i_left += 1
            else:
                _elements[_i_result] = _right_list[_i_right]
                _i_right += 1
            _i_result += 1
        # Добавляем в результирующий список "хвосты", оставшиеся от половинок.
        match (_i_left < len(_left_list), _i_right < len(_right_list)):
            case (True, False):
                _elements[_i_result:] = _left_list[_i_left:]
            case (False, True):
                _elements[_i_result:] = _right_list[_i_right:]

    return _elements


# --------------------------------------------------------------------------------------------
class GetRangesSort:
    """
    Вспомогательный класс для функции sort_by_shell(). Реализует различные методы формирования
    диапазонов чисел для перестановки. Класс является как итератором, так и классом со свойствами.
    Реализованы следующие методы:
    - Классический метод Shell
    - Hibbard
    - Sedgewick
    - Knuth
    - Fibonacci
    """

    def __init__(self, list_len: int, method: str = "Shell") -> None:
        self.__list_len: int = list_len
        self.__method: str = method.lower()
        self.__range: int = 0
        self.__i: int = 0
        self.__calc_res: list[int] = list()

        match self.__method:
            case "hibbard":
                _i = 1
                while (_res := (2**_i - 1)) <= self.__list_len:
                    self.__calc_res.append(_res)
                    _i += 1
            case "sedgewick":
                _i = 0
                while (_res := self.__get_sedgewick_range(_i)) <= self.__list_len:
                    self.__calc_res.append(_res)
                    _i += 1
            case "knuth":
                _i = 1
                while (_res := ((3**_i - 1) // 2)) <= (self.__list_len // 3):
                    self.__calc_res.append(_res)
                    _i += 1
            case "fibonacci":
                _i = 1
                while (_res := self.__get_fibonacci_range(_i)) <= self.__list_len:
                    self.__calc_res.append(_res)
                    _i += 1
            case "shell" | _:
                _res = self.__list_len
                while (_res := (_res // 2)) > 0:
                    self.__calc_res.append(_res)
                else:
                    self.__calc_res.sort()

        self.__i = len(self.__calc_res) - 1
        self.__range = self.__calc_res[self.__i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.getnextrange > 0:
            return self.__range
        else:
            raise StopIteration

    def __get_sedgewick_range(self, i: int) -> int:
        if i % 2 == 0:
            return 9 * (2**i - 2 ** (i // 2)) + 1
        else:
            return 8 * 2**i - 6 * 2 ** ((i + 1) // 2) + 1

    def __get_fibonacci_range(self, i: int) -> int:
        return (self.__get_fibonacci_range(i - 2) + self.__get_fibonacci_range(i - 1)) if i > 1 else 1

    @property
    def getnextrange(self) -> int:
        if self.__i >= 0:
            self.__range = self.__calc_res[self.__i]
            self.__i -= 1
        else:
            self.__range = 0

        return self.__range

    @property
    def getcurrentrange(self) -> int:
        return self.__range


def sort_by_shell(elements: Iterable, *, revers: bool = False, method: str = "Shell") -> list:
    """
    Функция сортировки методом Shell. Кроме классического метода формирования
    дипазанона чисел для перестановки, возможно использовать следующие методы:
    - Hibbard
    - Sedgewick
    - Knuth
    - Fibonacci

    Реализована двунаправленная сортировка.

    Args:
        elements (list): Список данных для сортировки

        revers (bool, optional): Если задано True, то сортировка по убыванию.. Defaults to False.

        method (str, optional): Мотод формирования диапазона: Shell, Hibbard, Sedgewick, Knuth,
        Fibonacci. Defaults to "Shell".

    Returns:
        list: Отсортированный список.
    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _elements: list = list(elements)
    except (ValueError, TypeError):
        return []

    _sort_order: int = -1 if revers else 1
    _ranges = GetRangesSort(len(_elements), method)
    for _range in _ranges:
        for _i_range in range(_range, len(_elements)):
            _i_current: int = _i_range
            while (_i_current >= _range) and (
                (_sort_order * _elements[_i_current]) < (_sort_order * _elements[_i_current - _range])
            ):
                _elements[_i_current], _elements[_i_current - _range] = (
                    _elements[_i_current - _range],
                    _elements[_i_current],
                )
                _i_current -= _range

    return _elements


# -------------------------------------------------------------------------------------------------
def sort_by_selection(elements: Iterable, *, revers: bool = False) -> list:
    """
    Функция сортировки методом выбора. Это улучшенный вариант пузырьковой сортировки,
    за счет сокращения числа перестановок элементов. Элементы переставляются не на
    каждом шаге итерации, а только лишь в конце текущей итерации. Дополнительно к
    классическому алгоритму добавлена возможность одновременного поиска максимального
    и минимального элементов текущего диапазона за одну итерацию. Реализована
    двунаправленная сортировка списка данных.

    Args:
        elements (list): Список данных для сортировки.
        revers (bool, optional): Если задано True, список сортируется по убыванию. Defaults to False.

    Returns:
        list: Возвращаемый отсортированный список.
    """
    # создаем копию передаваемого списка, дабы не влиять на оригинальный список
    try:
        _elements: list = list(elements)
    except (ValueError, TypeError):
        return []

    # Стартуем с дипазална равного длине списка данных, кроме последнего элемента.
    i_start: int = 0
    i_end: int = len(_elements) - 1
    # Потенциальные минимум и максимум в начале и конце диапазона
    i_min: int = i_start
    i_max: int = i_end
    _sort_order: int = -1 if revers else 1  # Задаем порядок сортировки
    # Перебираем диапазоны, сокращая длину каждого следующего диапазона на 2
    while i_start < i_end:
        # Т.к. до последнего элемента не доходим, необходимо перед итерацией
        # сравнить последний элемент с первым. Возмоно последний элемент
        # потенуиальный минимум текущего диапазона
        if (_sort_order * _elements[i_end]) < (_sort_order * _elements[i_start]):
            # Меняем местами первый и последний элементы текущего диапазона
            _elements[i_start], _elements[i_end] = _elements[i_end], _elements[i_start]
        for i_current in range(i_start, i_end, 1):
            # Если текущий элемент больше последнего в диапазоне, то это потенциальный максимум
            # для текущего дипазона.
            if (_sort_order * _elements[i_current]) > (_sort_order * _elements[i_max]):
                i_max = i_current
            # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
            elif (i_current > i_start) and (_sort_order * _elements[i_current]) < (_sort_order * _elements[i_min]):
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
    pass
