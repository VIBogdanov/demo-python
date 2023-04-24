from itertools import islice
from multiprocessing import Pool, cpu_count


def get_positive_int(value) -> int:
    """
    Вспомогательная функция. Проверяет значение на положительное целое.
    Если переданное значение не целое число, вернет ноль.
    Отрицательное число конвертирует в положительное.

    Args:
        value (_type_): Значение для проверки

    Returns:
        int: Возвращает целое положительное число
    """
    result: int = 0

    try:
        result = int(value)
    except (ValueError, TypeError):
        result = 0

    if result < 0:
        result *= -1

    return result


def get_ranges(list_len: int, range_len: int):
    """
    Вспомогательная функция-генератор, формирующая список диапазонов,
    на которые можно разбить исходноый список.

    Args:
        list_len (int): Длина исходного списка.
        range_len (int): Размер диапазона.

    Yields:
        _type_: Возвращает кортеж с начальной и конечной позицией.
    """
    i: int = 0
    # Корректируем возможные ошибки во входных параметрах
    _list_len: int = get_positive_int(list_len)
    _range_len: int = get_positive_int(range_len)
    # Исключаем ошибки и слишком короткие величины длин.
    if any(
        (
            (_range_len >= _list_len),
            (_list_len < 1),
            (_range_len < 1),
        )
    ):
        yield (0, _list_len)
    else:
        while (_range_end := _range_len * (i + 1)) <= _list_len:
            yield ((i * _range_len), _range_end)
            i += 1
        # В случае, если список не делится на диапазона без остатка, добавляем "хвост"
        if (_range_end := _list_len % _range_len) > 0:
            yield ((i * _range_len), (i * _range_len + _range_end))

    return


def _is_srt(elements, revers: bool = False) -> bool:
    """
    Вспомогательная функция, поэлементно проверяющая отсортирован ли исходный список
    в зависимости от заданного направления сортировки. При первом ложном сравнении
    итерация прерывается.

    Args:
        elements (_type_): Исходный список для проверки.
        revers (bool, optional): Направление сортировки.. Defaults to False.

    Returns:
        bool: True - список отсортирован.
    """
    _ln = len(elements)
    if _ln < 2:
        return True
    _sort_order: int = -1 if revers else 1
    # Альтернативный вариант чуть быстрее, но требует библиотечной функции islice
    # from itertools import islice
    for _current, _next in zip(elements, islice(elements, 1, None)):
        if (_sort_order * _current) > (_sort_order * _next):
            return False
    return True
    # for i in range(1, _ln):
    #    if (_sort_order * elements[i - 1]) > (_sort_order * elements[i]):
    #        return False
    # return True


def is_sorted(elements, revers: bool = False, chunksize: int | None = None) -> bool:
    """
    Проверяет отсортирован ли список. В случае больших списков используются
    параллельные вычисления. Для параллельных вычислений задается размер диапазонов,
    на котрые разбивается исходный список. Каждый диапазон проверяется в отдельном
    процессе. При проверке учитывается порядок сортировки.

    Args:
        elements (_type_): Список для проверки.
        revers (bool, optional): Порядок сортировки. Defaults to False.
        chunksize (int | None): Размер диапазона, на который можно разбить список. Defaults to None.

    Returns:
        bool: True, если список отсортирован.
    """
    margs_list: list = list()  # массив значений для параметров функции в режиме много задачности
    _ln = len(elements)
    # Пустые списки или списки из одного элемента всегда отсортированы
    if _ln < 2:
        return True

    # размер минимального диапазона, на которые делится исходный список
    _range_size: int = get_positive_int(chunksize)
    # Если размер диапазона не задан, вычисляем исходя из производительности CPU
    if _range_size == 0:
        # Считаем, что частота процессора 4000
        _range_size = max(_ln // cpu_count(), cpu_count() * 4000)

    # Если исходный список можно разделить хотя бы на 2 подсписка
    if _ln >= (_range_size * 2):
        # Разбиваем исходный список на диапазоны и проверяем каждый диапазон в отдельном процессе.
        # Для каждого диапазона проверяем пограничный элемент на признак сортировки.
        # Возможна ситуация, когда два отдельный подсписка отсортированы, но целый список нет
        # Например: [1,2,4,3,5,6]. Если разделить на два, то оба подсписка будут отсортированы,
        # но при этом исходный список не отсортирован.
        for i_start, i_end in get_ranges(_ln, _range_size):
            if (i_end < _ln) and (elements[i_end - 1] > elements[i_end] or elements[i_end] > elements[i_end + 1]):
                return False
            # Формируем список параметров для вызова функции проверки сортировки
            # в многозадачном режиме для различных диапазонов
            margs_list.append(tuple([elements[i_start:i_end], revers]))
        # Запускам пул параллельных процессов для проверки сортировки набора диапазонов
        with Pool() as mpool:
            # Возвращаем True, только если все без исключения проверки успешны
            return all(mpool.starmap_async(_is_srt, margs_list).get())
    else:
        # Для небольших списков нет смысла использовать многозадачность
        return _is_srt(elements, revers)


if __name__ == "__main__":
    pass
