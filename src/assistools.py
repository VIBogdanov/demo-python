from itertools import pairwise
from multiprocessing import Pool, cpu_count
from typing import Generator, Iterable

CPU_FREQUENCY = 4000  # Считаем, что частота процессора 4000


def get_positive_int(value) -> int:
    """
    Проверяет значение на положительное целое.
    Если переданное значение невозможно представить как целое число,
    вернет ноль. Отрицательное число конвертирует в положительное.

    Args:
        value (_type_): Значение для проверки

    Returns:
        int: Возвращает целое положительное число
    """
    _result: int = 0
    try:
        _result = int(value)
    except (ValueError, TypeError):
        _result = 0
    else:
        if _result < 0:
            _result *= -1
    finally:
        return _result


def get_ranges_index(list_len: int, range_len: int) -> Generator[tuple[int, int], None, None]:
    """
    Функция-генератор, формирующая список индексов диапазонов,
    на которые можно разбить исходноый список длиной list_len.

    Args:
        list_len (int): Длина исходного списка.
        range_len (int): Размер диапазона.

    Yields:
        _type_: Возвращает кортеж с начальным и конечным индексами диапазона.
    """
    # Корректируем возможные ошибки во входных параметрах
    _list_len: int = get_positive_int(list_len)
    _range_len: int = get_positive_int(range_len)
    # Исключаем ошибки и слишком короткие величины длин.
    if any(
        (
            (_list_len < 1),
            (_range_len < 1),
        )
    ):
        yield (0, _list_len)
    else:
        for i in range(0, _list_len, _range_len):
            yield (i, i + _range_len) if (i + _range_len) < _list_len else (i, _list_len)


# Обертка для _is_srt, чтобы можно было передавать кортеж с аргументами
# в качестве единственного параметра в функции imap_unordered (см. is_sorted)
def _is_srt_imap_unordered(_args: tuple) -> bool:
    return _is_srt(*_args)


def _is_srt(elements: Iterable, revers: bool = False) -> bool:
    """
    Вспомогательная функция, поэлементно проверяющая отсортирован ли исходный список
    в зависимости от заданного направления сортировки. При первом ложном сравнении
    итерация прерывается.

    Args:
        range1 (_type_): Исходный список для проверки без последнего элемента.

        range1 (_type_): Исходный список для проверки без первого элемента.

        revers (bool, optional): Направление сортировки. Defaults to False.

    Returns:
        bool: True - список отсортирован.
    """
    _sort_order: int = -1 if revers else 1

    for _current, _next in pairwise(elements):
        if (_sort_order * _current) > (_sort_order * _next):
            return False

    return True


def is_sorted(elements, revers: bool = False, rangesize: int | None = None) -> bool:
    """
    Проверяет отсортирован ли список. В случае больших списков используются
    параллельные вычисления. Для параллельных вычислений задается размер диапазонов,
    на котрые разбивается исходный список. Каждый диапазон проверяется в отдельном
    процессе. При проверке учитывается порядок сортировки.

    Args:
        elements (_type_): Список для проверки.
        revers (bool, optional): Порядок сортировки. Defaults to False.
        rangesize (int | None): Размер диапазона, на который можно разбить список. Defaults to None.

    Returns:
        bool: True, если список отсортирован.
    """
    # Пустые списки или списки из одного элемента всегда отсортированы
    if (_ln := len(elements)) < 2:
        return True

    _cpu: int = cpu_count()
    _result: bool = True  # По умолчанию считаем список отсортированным

    # размер диапазонов, на которые делится исходный список
    _range_size: int = get_positive_int(rangesize)
    # Если размер диапазона не задан, вычисляем исходя из производительности CPU
    if _range_size == 0:
        _range_size = _cpu * CPU_FREQUENCY

    _ranges_count: int = _ln // _range_size + (1 if _ln % _range_size > 0 else 0)

    # Если исходный список можно разделить хотя бы на 2 подсписка
    # запускаем многозадачную обработку
    if _ranges_count > 1:
        # Разбиваем исходный список на диапазоны и проверяем каждый диапазон в отдельном процессе.
        # Для каждого диапазона (кроме последнего) сравниваем последний элемент с первым элементом
        # следующего диапазона, для чего увеличиваем конечный индекс диапазона на 1.
        # Возможна ситуация, когда два отдельный подсписка отсортированы, но целый список нет
        # Например: [1,2,4,3,5,6]. Если разделить пополам, то оба подсписка будут отсортированы,
        # но при этом исходный список не отсортирован.
        _margs_list = (
            (iter(elements[i_start:(i_end + 1 if i_end < _ln else i_end)]), revers)
            for i_start, i_end in get_ranges_index(_ln, _range_size)
        )

        # Запускаем пул параллельных процессов для проверки сортировки набора диапазонов
        # - результаты получаем сразу по готовности не дожидаясь завершения всех проверок
        # - возможно досрочное завершение обработки результатов
        with Pool(processes=min(_cpu, _ranges_count)) as mpool:
            # Загружаем задачи в пул и запускаем итератор для получения результатов
            for _result in mpool.imap_unordered(_is_srt_imap_unordered, _margs_list):
                # Если один из результатов False, останавливаем цикл получения результатов
                if _result is False:
                    # Отменяем выполнение задач, которые еще не загружены в пул
                    mpool.terminate()
                    break  # Прерывает цикл (for) проверки результатов

        return _result
    else:
        # Для небольших списков нет смысла использовать многозадачность
        return _is_srt(iter(elements), revers)


if __name__ == "__main__":
    pass
