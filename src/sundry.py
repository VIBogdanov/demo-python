from collections import defaultdict
from functools import cache, reduce
from itertools import accumulate
from multiprocessing import Pool
from typing import Any


# ------------------------------------------------------------------------------
def find_intervals(
    elements: list[int],
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


# -------------------------------------------------------------------------------
def find_nearest_number(
    number: int | str,
    previous: bool = True,
    multiproc: bool = False,
) -> int | None:
    """
    Функция поиска ближайшего целого числа, которое меньше или больше заданного
    и состоит из тех же цифр.

    Args:
        number (int | str): Целое число, относительнго которого осуществляется
        поиск. Допускается строковое представление числа, положительные или
        отрицательные значения.

        previous (bool, optional): Направление поиска: ближайшее меньшее или
        большее. По-умолчанию ближайшее меньшее.

        multiproc (bool, optional): Использование многозадачности при поиске.
        По-умолчанию отключено.

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

    result: int | None = None  # по-умолчанию, в случае безуспешного поиска, возвращаем None
    search_direction: int = 1 if previous else -1  # направление поиска: большее или меньшее
    sign_number: int = 1

    if input_number < 0:  # если входное число отрицательное
        sign_number = -1  # сохраняем знак числа
        input_number *= -1  # переводим входное число на положительное значение
        search_direction *= -1  # меняем направление поиска
    # массив цифр из входного числа
    digits_list: list[int] = [int(digit) for digit in str(input_number)]
    # списки margs и mres используются в режиме многозадачности
    margs_list: list = list()  # массив значений для параметров функции в режиме многозадачности
    mres_list: tuple = tuple()  # список результатов, полученных от многозадачной функции
    # цикл перебора цифр входного числа справа на лево (с хвоста к голове) кроме первой цифры
    for i in range(len(digits_list) - 1, 0, -1):
        if multiproc:  # если включен режим многозадачности
            # сохраняем наборы входных значений в массиве
            # передаем копию массива цифр входного числа вместо ссылки,
            # чтобы не влиять на исходный массив при перестановке цифр внутри подпрограммы
            margs_list.append(tuple([digits_list.copy(), i, search_direction]))
        else:
            # в синхронном режиме последовательно вызываем подпрограмму поиска большего
            # или меньшего числа в зависимости от направления поиска
            found_number = _do_find_nearest(digits_list.copy(), i, search_direction)
            if found_number is not None:
                # если это первое найденное число (возможно единственное), сохраняем его как
                # результирующее и переходим к следующей цифре
                if result is None:
                    result = found_number
                else:
                    # сравниваем очередное найденное число с ранее сохраненным и выбираем большее
                    # или меньшее из них в зависимости от направления поиска
                    result = max(result, found_number) if search_direction == 1 else min(result, found_number)
    # при включенном режиме многозадачности формируем пул процессов и передаем в него
    # подпрограмму с набором различных параметров для параллельного запуска
    if multiproc:
        with Pool() as mpool:
            # из возвращаемых результирующих чисел исключаем значения равные None
            mres_list = tuple(
                number_value for number_value in mpool.starmap(_do_find_nearest, margs_list) if number_value is not None
            )
        if mres_list:
            # если список результирующих чисел не пуст, находим наибольшее или наименьшее число
            # в зависимости от направления поиска
            result = max(mres_list) if search_direction == 1 else min(mres_list)
    # если искомое число найдено и входное число было отрицательным, восстанавливаем знак минус
    if result is not None and sign_number == -1:
        result *= -1

    return result


def _do_find_nearest(
    digits_list: list[int],
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
        digits_list (list[int]): Массив цифр исходного числа

        current_index (int): Текущая позиция исходного числа

        search_direction (int): Направление поиска: ближайшее большее или меньшее

    Returns:
        (int | None): Возвращает найденное целое число или None в случае
        безуспешного поиска

    """
    i: int = current_index  # текущая позиция исходного числа, относительно которой ведется поиск
    for k in range(i - 1, -1, -1):  # просматриваем все цифры левее текущей позиции
        # сравниваем с текущей позицией, учитывая направление поиска
        if (search_direction * digits_list[k]) > (search_direction * digits_list[i]):
            # в случае успешного сравнения, переставляем местами найденную цифру с текущей
            digits_list[k], digits_list[i] = digits_list[i], digits_list[k]
            # если первая цифра полученного числа после перестановки не равна 0,
            # выполняем сортировку правой части числа
            if digits_list[0] > 0:
                k += 1  # правая часть числа начинается со сдвигом от найденной позиции
                # сортируем правую часть числа (по возрвстанию или по убыванию) с учетом направления поиска
                digits_list[k::] = sorted(digits_list[k::], reverse=(search_direction == 1))
                # собираем из массива цифр результирующее число
                return reduce(lambda dig_prev, dig_next: 10 * dig_prev + dig_next, digits_list)
    return None


# ------------------------------------------------------------------------------------
def find_item_by_binary(
    elements: list | tuple,
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
    _is_forward = True  # По умолчанию считаем входной массив отсортированным по возрастанию
    if elements[0] > elements[-1]:
        _is_forward = False  # Иначе по убыванию
    # Стартуем с первого и последнего индекса массива
    i_first = 0
    i_last = len(elements) - 1

    i_target = None  # Возвращаемый индекс найденого значения

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
def sort_by_bubble(elements: list, revers: bool = False) -> list:
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
    i_start = 0
    i_end = len(elements) - 1
    _sort_order = -1 if revers else 1  # Задаем порядок сортировки

    while i_start < i_end:
        for i_current in range(i_start, i_end, 1):
            # Если текущий элемент больше следующего, то переставляем их местами. Это потенциальный максимум.
            if (_sort_order * elements[i_current]) > (_sort_order * elements[i_current + 1]):
                elements[i_current], elements[i_current + 1] = elements[i_current + 1], elements[i_current]
                # Одновременно проверяем на потенциальный минимум, сравнивая с первым элементом текущего диапазона.
                if (_sort_order * elements[i_current]) < (_sort_order * elements[i_start]):
                    elements[i_start], elements[i_current] = elements[i_current], elements[i_start]
        # После каждой итерации по элементам списка, сокращаем длину проверяемого диапазона на 2,
        # т.к. на предыдущей итерации найдены одновременно минимум и максимум
        i_start += 1
        i_end -= 1

    return elements


# ------------------------------------------------------------------------------------------------
def sort_by_merge(elements: list, revers: bool = False) -> list:
    """
    Функция сортировки методом слияния. Поддерживается сортировка как по возрастанию,
    так и по убыванию.

    Args:
        elements (list): Список данных для сортировки.
        revers (bool, optional): Если задано True, то сортировка по убыванию. Defaults to False.

    Returns:
        list: Результирующий отсортированный список.
    """
    if len(elements) > 1:
        # Делим исходный список пополам.
        _i_middle: int = len(elements) // 2
        # Рекурсивно вызываем функцию до тех пор,
        # пока исходный список не будет разложен поэлементно.
        _left_list: list = sort_by_merge(elements[:_i_middle], revers)
        _right_list: list = sort_by_merge(elements[_i_middle:], revers)
        # Собираем список из стека рекурсивных вызовов
        _i_left: int = 0
        _i_right: int = 0
        _i_result: int = 0
        _sort_order: int = -1 if revers else 1  # Учитываем порядок сортировки
        # Сравниваем поэлементно половинки списка и добавляем в результирующий список
        # меньший или больший элемент, в зависимости от порядка сортировки.
        while _i_left < len(_left_list) and _i_right < len(_right_list):
            if (_sort_order * _left_list[_i_left]) < (_sort_order * _right_list[_i_right]):
                elements[_i_result] = _left_list[_i_left]
                _i_left += 1
            else:
                elements[_i_result] = _right_list[_i_right]
                _i_right += 1
            _i_result += 1
            # Добавляем в результирующий список "хвосты", оставшиеся от половинок.
            match (_i_left < len(_left_list), _i_right < len(_right_list)):
                case (True, False):
                    elements[_i_result:] = _left_list[_i_left:]
                    _i_result = len(elements)
                case (False, True):
                    elements[_i_result:] = _right_list[_i_right:]
                    _i_result = len(elements)
    # Следует учесть, что меняется исходный список данных и возвращается его отсортированная версия.
    return elements


# --------------------------------------------------------------------------------------------
class GetRangeSort:
    """
    Вспомогательный класс для функции sort_by_shell(). Реализует различные методы формарования
    диапазона чисел для перестановки. Реализованы следующие методы:
    - Классический метод Shell
    - Hibbard
    - Sedgewick
    - Knuth
    - Fibonacci
    """
    def __init__(self, list_len: int, method: str = 'Shell') -> None:
        self.__len: int = list_len
        self.__method: str = method.lower()
        self.__range: int = None
        self.__i: int = 0
        match self.__method:
            case 'hibbard':
                while self.__get_hibbard_range(self.__i) <= self.__len:
                    self.__i += 1
                else:
                    self.__i -= 1
            case 'sedgewick':
                while self.__get_sedgewick_range(self.__i) < self.__len:
                    self.__i += 1
                else:
                    self.__i -= 1
            case 'knuth':
                while self.__get_knuth_range(self.__i) < (self.__len // 3):
                    self.__i += 1
            case 'fibonacci':
                while self.__get_fibonacci_range(self.__i) <= self.__len:
                    self.__i += 1
                else:
                    self.__i -= 1
            case 'shell' | _:
                self.__range = None
    
    @cache
    def __get_hibbard_range(self, i: int) -> int:
        return (2**i - 1)

    @cache
    def __get_sedgewick_range(self, i: int) -> int:
        if i % 2 == 0:
            return 9 * (2**i - 2**(i//2)) + 1
        else:
            return 8 * 2**i - 6 * 2**((i+1)//2) + 1
        
    @cache
    def __get_knuth_range(self, i: int) -> int:
        return (3**i - 1) // 2
    
    @cache
    def __get_fibonacci_range(self, i: int) -> int:
        return (self.__get_fibonacci_range(i-2) + self.__get_fibonacci_range(i-1)) if i > 1 else 1

    @property
    def nextrange(self) -> int:
        match self.__method:
            case 'hibbard':
                if self.__i > 0:
                    self.__range = self.__get_hibbard_range(self.__i)
                    self.__i -= 1
                else:
                    self.__range = 0
            case 'sedgewick':
                if self.__i >= 0:
                    self.__range = self.__get_sedgewick_range(self.__i)
                    self.__i -= 1
                else:
                    self.__range = 0
            case 'knuth':
                if self.__i > 0:
                    self.__range = self.__get_knuth_range(self.__i)
                    self.__i -= 1
                else:
                    self.__range = 0
            case 'fibonacci':
                if self.__i > 0:
                    self.__range = self.__get_fibonacci_range(self.__i)
                    self.__i -= 1
                else:
                    self.__range = 0
            case 'shell' | _:
                self.__range = (self.__len // 2) if self.__range is None else (self.__range // 2)
        return self.__range
    
    @property
    def getrange(self) -> int:
        return 0 if self.__range is None else self.__range


def sort_by_shell(elements: list, revers: bool = False, method: str = "Shell") -> list:
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

        method (str, optional): Мотод формирования диапазона: Shell, Hibbard, Sedgewick, Knuth, Fibonacci. Defaults to "Shell".

    Returns:
        list: Отсортированный список.
    """
    _sort_order: int = -1 if revers else 1
    _range = GetRangeSort(len(elements), method)
    while _range.nextrange > 0:
        for _i_range in range(_range.getrange, len(elements)):
            _i_current: int = _i_range
            while (_i_current >= _range.getrange) and (
                (_sort_order * elements[_i_current]) < (_sort_order * elements[_i_current - _range.getrange])
            ):
                elements[_i_current], elements[_i_current - _range.getrange] = (
                    elements[_i_current - _range.getrange],
                    elements[_i_current],
                )
                _i_current -= _range.getrange
    return elements


if __name__ == "__main__":
    pass
