from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

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
    result: int = 0

    try:
        result = int(value)
    except (ValueError, TypeError):
        result = 0
    else:
        if result < 0:
            result *= -1
    finally:
        return result


def get_ranges(list_len: int, range_len: int):
    """
    Функция-генератор, формирующая список диапазонов,
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
            (_list_len < 1),
            (_range_len < 1),
        )
    ):
        yield (0, _list_len)
    else:
        for i in range(0, _list_len, _range_len):
            yield (i, i + _range_len) if (i + _range_len) < _list_len else (i, _list_len)


def _is_srt(range1, range2, revers: bool = False) -> bool:
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

    for _current, _next in zip(range1, range2):
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

    _margs_list: list[tuple] = list()  # массив значений для параметров функции в режиме много задачности
    _cpu: int = cpu_count()
    _result: bool = True  # По умолчанию считаем список отсортированным

    # размер минимального диапазона, на которые делится исходный список
    _range_size: int = get_positive_int(rangesize)
    # Если размер диапазона не задан, вычисляем исходя из производительности CPU
    if _range_size == 0:
        _range_size = _cpu * CPU_FREQUENCY

    # Если исходный список можно разделить хотя бы на 2 подсписка
    # запускаем многозадачную обработку
    if _ln >= (_range_size * 2):
        # Разбиваем исходный список на диапазоны и проверяем каждый диапазон в отдельном процессе.
        # Для каждого диапазона проверяем пограничный элемент на признак сортировки.
        # Возможна ситуация, когда два отдельный подсписка отсортированы, но целый список нет
        # Например: [1,2,4,3,5,6]. Если разделить на два, то оба подсписка будут отсортированы,
        # но при этом исходный список не отсортирован.
        for i_start, i_end in get_ranges(_ln, _range_size):
            if (i_end < _ln - 1) and (elements[i_end - 1] > elements[i_end] or elements[i_end] > elements[i_end + 1]):
                return False
            # Формируем список параметров для вызова функции проверки сортировки
            # в многозадачном режиме для различных диапазонов
            # Вместо списка передаем итераторы со смещением, дабы избежать
            # ощутимого расхода памяти при больших исходных данных
            _margs_list.append(
                tuple(
                    (
                        iter(elements[i_start:(i_end - 1)]),
                        iter(elements[(i_start + 1):i_end]),
                        revers,
                    )
                )
            )
        # Запускаем пул параллельных процессов для проверки сортировки набора диапазонов
        # Главные его преимущества:
        # - результаты получаем сразу по готовности не дожидаясь завершения всех задач
        # - досрочное завершение обработки результатов
        with ProcessPoolExecutor(max_workers=min(_cpu, len(_margs_list))) as _executor:
            # Используем генератор, который работает быстрее и память экономит
            _futures = (_executor.submit(_is_srt, *_marg) for _marg in _margs_list)
            # Запускаем цикл получения результатов по мере их поступления
            for _future in as_completed(_futures):
                # Если хотя бы одна из частей списка не отсортирована
                if (_result := _future.result()) is False:
                    # Останавливаем дальнейшую загрузку задач в пул процессов
                    _executor.shutdown(wait=False, cancel_futures=True)
                    break  # Прерывает цикл (for) проверки результатов
        return _result
    else:
        # Для небольших списков нет смысла использовать многозадачность
        return _is_srt(iter(elements[:-1]), iter(elements[1:]), revers)


if __name__ == "__main__":
    pass
