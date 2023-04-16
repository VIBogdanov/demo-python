from functools import reduce
from multiprocessing import Pool
from typing import Union


def find_nearest_number(
    number: int | str,
    previous: bool = True,
    multiproc: bool = False,
) -> int | None:
    """
    Функция поиска ближайшего целого числа, которое меньше или больше исходного
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

    result: Union[int, None] = None  # по-умолчанию, в случае безуспешного поиска, возвращаем None
    search_direction: int = 1 if previous else -1  # направление поиска: большее или меньшее
    sign_number: int = 1

    if input_number < 0:  # если входное число отрицательное
        sign_number = -1  # сохраняем знак числа
        input_number *= -1  # переводим входное число на положительное значение
        search_direction *= -1  # меняем направление поиска
    # массив цифр из входного числа
    digits_list: list[int] = [int(digit) for digit in str(input_number)]
    # списки margs и mres используются в режиме многозадачности
    margs_list = list()  # массив значений для параметров функции в режиме многозадачности
    mres_list = list()  # список результатов, полученных от многозадачной функции
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
            mres_list = [
                number_value for number_value in mpool.starmap(_do_find_nearest, margs_list) if number_value is not None
            ]
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
) -> Union[int, None]:
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


if __name__ == "__main__":
    print(find_nearest_number(273145))
