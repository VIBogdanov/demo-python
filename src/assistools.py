from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from multiprocessing import Pool, cpu_count
from typing import Any, NamedTuple, TypeAlias, TypeVar

CPU_FREQUENCY = 4000  # Считаем, что частота процессора 4000
TAny = TypeVar("TAny")
NumberStrNone: TypeAlias = int | float | str | None


def is_even(n: int) -> bool:
    """
    Крайне простой алгоритм проверки целого числа на четность.
    У четных целых первый бит всегда равен 0.
    Не требуется получение остатка делением на два: n % 2

    Args:
        n (int): Целое число. Допускается положительные и отрицательные числа.

    Returns:
        bool: True - если число четное.
    """
    return not n & 1


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
        result: int = int(value)
    except (ValueError, TypeError):
        result = 0

    return abs(result)


class RangeIndex(NamedTuple):
    start: int
    end: int


def get_ranges_index(list_len: int, range_len: int) -> Iterator[RangeIndex]:
    """
    Функция-генератор, формирующая список индексов диапазонов заданной длины,
    на которые можно разбить исходный список длиной list_len.

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
            yield (
                RangeIndex(i, i + _range_len)
                if (i + _range_len) < _list_len
                else RangeIndex(i, _list_len)
            )


def _is_srt(args: tuple[Iterator, bool]) -> bool:
    """
    Вспомогательная функция, поэлементно проверяющая отсортирован ли исходный список
    в зависимости от заданного направления сортировки. При первом ложном сравнении
    итерация прерывается.

    Args:
        args (tuple[Iterable, bool]): Кортеж параметров - итератор списока для проверки и
        направление сортировки

    Returns:
        bool: True/False - список отсортирован / не отсортирован.
    """
    elements, is_revers = args

    try:
        _prev = next(elements)
    except StopIteration:
        return True

    for _next in elements:
        if (_next > _prev) if is_revers else (_prev > _next):
            return False
        _prev = _next

    return True


def is_sorted(
    elements: Sequence[TAny],
    *,
    revers: bool = False,
    rangesize: int | None = None,
) -> bool:
    """
    Проверяет отсортирован ли список. В случае больших списков используются
    параллельные вычисления. Для параллельных вычислений задается размер диапазонов,
    на которые разбивается исходный список. Каждый диапазон проверяется в отдельном
    процессе. При проверке учитывается порядок сортировки.

    Args:
        elements (Sequence): Массив данных для проверки.

        revers (bool, optional): Порядок сортировки. Defaults to False.

        rangesize (int | None): Размер диапазона, на который можно разбить список. Defaults to None.

    Returns:
        bool: True, если список отсортирован.
    """
    # Пустые списки или списки из одного элемента всегда отсортированы
    if (ln := len(elements)) < 2:
        return True

    cpu: int = cpu_count()
    result: bool = True  # По умолчанию считаем список отсортированным

    # Если размер диапазона не задан, вычисляем исходя из производительности CPU
    if (range_size := get_positive_int(rangesize)) == 0:
        range_size = cpu * max(round(ln**0.5), CPU_FREQUENCY)

    ranges_count: int = ln // range_size + int(bool(ln % range_size))

    # Если исходный список можно разделить хотя бы на 2 подсписка
    # запускаем многозадачную обработку
    if ranges_count > 1:
        # Разбиваем исходный список на диапазоны и проверяем каждый диапазон в отдельном процессе.
        # Для каждого диапазона (кроме последнего) сравниваем последний элемент с первым элементом
        # следующего диапазона, для чего увеличиваем конечный индекс диапазона на 1.
        # Возможна ситуация, когда два отдельный подсписка отсортированы, но целый список нет
        # Например: [1,2,4,3,5,6]. Если разделить пополам, то оба подсписка будут отсортированы,
        # но при этом исходный полный список не отсортирован.
        margs_list = (
            (iter(elements[i_start : (i_end + int(i_end < ln))]), revers)  # noqa: E203
            for i_start, i_end in get_ranges_index(ln, range_size)
        )

        # Запускаем пул параллельных процессов для проверки сортировки набора диапазонов
        # - результаты получаем сразу по готовности не дожидаясь завершения всех проверок
        # - возможно досрочное завершение обработки результатов
        with Pool(processes=min(cpu, ranges_count)) as mpool:
            # Загружаем задачи в пул и запускаем итератор для получения результатов по мере готовности
            for result in mpool.imap_unordered(_is_srt, margs_list):
                # Если один из результатов False, останавливаем цикл получения результатов
                if not result:
                    # Отменяем выполнение задач, которые еще не загружены в пул
                    mpool.terminate()
                    break  # Прерывает цикл (for) проверки результатов

        return result
    else:
        # Для небольших списков нет смысла использовать многозадачность
        return _is_srt((iter(elements), revers))


# -------------------------------------------------------------------------------------------------
def get_day_week_index(iday: int, imonth: int, iyear: int) -> int:
    if m := abs(imonth) % 12:
        imonth = m
    else:
        imonth = 12

    iyear = abs(iyear)
    # По древнеримскому календарю год начинается с марта.
    # Январь и февраль относятся к прошлому году
    if imonth == 1 or imonth == 2:
        iyear -= 1
        imonth += 10
    else:
        imonth -= 2

    icentury: int = iyear // 100  # количество столетий
    iyear -= icentury * 100  # год в столетии

    # Original: (day + (13*month-1)/5 + year + year/4 + century/4 - 2*century + 777) % 7
    return int(
        (abs(iday) + (13 * imonth - 1) // 5 + (5 * iyear - 7 * icentury) // 4 + 777) % 7
    )


# -------------------------------------------------------------------------------------------------
def get_day_week_name(iday: int, imonth: int, iyear: int) -> str:
    dw: list[str] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    return dw[get_day_week_index(iday, imonth, iyear)]


# -------------------------------------------------------------------------------------------------
def is_includes_elements(data: Iterable[Any], pattern: Iterable[Any]) -> bool:
    """
    Проверяет вхождение одного набора данных в другой. Сортировка не требуется, порядок не важен,
    дублирование элементов допускается, относительный размер списков не принципиален.

    Args:
        data: Список, с которым сравнивается pattern.

        pattern: Проверяемый список на вхождение в data.

    Returns:
        bool: True - если все элементы из pattern присутствуют в data.
    """
    # Подсчитываем количество элементов в исходном списке
    cdata = Counter(data)
    # Вычитаем элементы проверяемого списка. При этом, если в исходном списке нет такого элемента,
    # то он будет добавлен со знаком минус.
    cdata.subtract(Counter(pattern))
    # Удаляем из словаря все элементы, количество которых больше или равно нулю
    # Если словарь становится пустым, то проверяемый список полностью содержится в исходном
    # Иначе в проверяемом списке есть элементы, которых нет в исходном, либо их меньше
    return len(-cdata) == 0


# -------------------------------------------------------------------------------------------------
def main():
    print(
        "\n- Формирует список индексов диапазонов, на которые можно разбить список заданной длины."
    )
    print(" get_ranges_index(50, 10) -> ", end="")
    for res in get_ranges_index(50, 10):
        print(tuple(res), end=" ")

    print("\n\n- Проверяет вхождение одного набора данных в другой.")
    print(
        f" is_includes_elements([1, 2, 3, 4, 5, 6, 7, 8, 8], [1, 2, 4, 8]) -> {is_includes_elements([1, 2, 3, 4, 5, 6, 7, 8, 8], [1, 2, 4, 8])}"
    )


# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from time import time

    data = range(100_000_000)
    start = time()
    res = is_sorted(data)
    print(f"Общее время выполнения is_sorted({res}):", time() - start)

    main()
