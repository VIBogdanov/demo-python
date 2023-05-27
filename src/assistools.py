from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise
from multiprocessing import Pool, cpu_count
from typing import NamedTuple, TypeAlias, TypeVar

CPU_FREQUENCY = 4000  # Считаем, что частота процессора 4000
T = TypeVar("T")
NumberStrNone: TypeAlias = int | float | str | None


def get_positive_int(value: NumberStrNone) -> int:
    """
    Проверяет значение на положительное целое.
    Если переданное значение невозможно представить как целое число,
    вернет ноль. Отрицательное число конвертирует в положительное.

    Args:
        value (NumberNone): Значение для проверки. Число или None

    Returns:
        int: Возвращает целое положительное число
    """
    if value is None:
        return 0

    try:
        _result: int = int(value)
    except (ValueError, TypeError):
        _result = 0

    return abs(_result)


class RangeIndex(NamedTuple):
    start: int
    end: int


def get_ranges_index(list_len: int, range_len: int) -> Iterator[RangeIndex]:
    """
    Функция-генератор, формирующая список индексов диапазонов заданной длины,
    на которые можно разбить исходноый список длиной list_len.

    Args:
        list_len (int): Длина исходного списка.

        range_len (int): Размер диапазона.

    Yields:
        Iterator[RangeIndex]: Возвращает кортеж с начальным и конечным индексами диапазона.
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
        yield RangeIndex(0, _list_len)
    else:
        for i in range(0, _list_len, _range_len):
            yield RangeIndex(i, i + _range_len) if (i + _range_len) < _list_len else RangeIndex(i, _list_len)


def _is_srt(args: tuple[Iterable, bool]) -> bool:
    """
    Вспомогательная функция, поэлементно проверяющая отсортирован ли исходный список
    в зависимости от заданного направления сортировки. При первом ложном сравнении
    итерация прерывается.

    Args:
        args (tuple[Iterable, bool]): Кортеж параметров - список для проверки и
        направление сортировки

    Returns:
        bool: True/False - список отсортирован / не отсортирован.
    """
    _elements, _revers = args
    for _current, _next in pairwise(_elements):
        if (_next > _current) if _revers else (_current > _next):
            return False

    return True


def is_sorted(
    elements: Sequence[T],
    *,
    revers: bool = False,
    rangesize: int | None = None,
) -> bool:
    """
    Проверяет отсортирован ли список. В случае больших списков используются
    параллельные вычисления. Для параллельных вычислений задается размер диапазонов,
    на котрые разбивается исходный список. Каждый диапазон проверяется в отдельном
    процессе. При проверке учитывается порядок сортировки.

    Args:
        elements (Sequence): Список для проверки.
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

    # Если размер диапазона не задан, вычисляем исходя из производительности CPU
    if (_range_size := get_positive_int(rangesize)) == 0:
        _range_size = _cpu * max(round(_ln**0.5), CPU_FREQUENCY)

    _ranges_count: int = _ln // _range_size + int(bool(_ln % _range_size))

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
            (iter(elements[i_start : (i_end + int(i_end < _ln))]), revers)  # noqa: E203
            for i_start, i_end in get_ranges_index(_ln, _range_size)
        )

        # Запускаем пул параллельных процессов для проверки сортировки набора диапазонов
        # - результаты получаем сразу по готовности не дожидаясь завершения всех проверок
        # - возможно досрочное завершение обработки результатов
        with Pool(processes=min(_cpu, _ranges_count)) as mpool:
            # Загружаем задачи в пул и запускаем итератор для получения результатов
            for _result in mpool.imap_unordered(_is_srt, _margs_list):
                # Если один из результатов False, останавливаем цикл получения результатов
                if _result is False:
                    # Отменяем выполнение задач, которые еще не загружены в пул
                    mpool.terminate()
                    break  # Прерывает цикл (for) проверки результатов

        return _result
    else:
        # Для небольших списков нет смысла использовать многозадачность
        return _is_srt((iter(elements), revers))


if __name__ == "__main__":
    from time import time
    data = range(10_000_000)
    start = time()
    res = is_sorted(data)
    print(f"Общее время выполнения is_sorted({res}):", time() - start)
    pass
