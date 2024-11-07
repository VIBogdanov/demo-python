import ctypes
import functools
import inspect
import itertools
import logging
import multiprocessing
import sys
from collections import Counter, OrderedDict, deque
from collections.abc import Generator, Iterable, Iterator, Sequence, Sized
from typing import Any

from demo.timers import MiniTimers

CPU_FREQUENCY = 4000  # Считаем, что частота процессора 4000


# ---------------------Decorators-------------------------------------------------------
def type_checking(*type_args, **type_kwargs):
    """Декоратор, позволяющий выполнять проверку типов аргументов, передаваемых в функцию.
    Требование проверки типов можно задавать как для всех аргументов, так и выборочно.
    Требуемые типы сопоставляются с аргументами либо попозиционно, либо как ключ-значение.
    Требуемый тип задается либо как отдельное значение, либо как кортеж типов.

    Examples:
        >>>
        @type_checking(int, (int, str), z=float)
        def somefunction(x, y, z=4.5):
            pass
        # Альтернативный вариант
        @type_checking(y = (int, str), x = int)
        def somefunction(x, y, z=4.5):
            pass
        # Результат работы декоратора
        somefunction(1, 3, z=123.5)   #OK
        somefunction(1, '3', z=123.5)   #OK
        somefunction(1, 3)   #OK
        somefunction('1', 3, z=123.5)   #Error
        somefunction(1, 3, z=123)   #Error for first variant
        somefunction(1, 3.4)   #Error
    """

    def decorate(func):
        # В режиме оптимизации отключаем декоратор
        if not __debug__:
            return func
        # Формируем словарь, связывающий арнументы функции с требованиями типов, заданными в декораторе
        func_signature: inspect.Signature = inspect.signature(func)
        args_types: OrderedDict[str, Any] = func_signature.bind_partial(
            *type_args, **type_kwargs
        ).arguments

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Формируем словарь с именами преданных функции аргументов и их значениями
            for arg_name, arg_value in func_signature.bind(
                *args, **kwargs
            ).arguments.items():
                # Если для данного аргумента задана проверка типа
                # И тип значения аргумента не соответствует заданному в декораторе
                if arg_name in args_types and not isinstance(
                    arg_value, arg_types := args_types[arg_name]
                ):
                    # Для join нужен кортеж
                    arg_types = (
                        arg_types if isinstance(arg_types, Iterable) else (arg_types,)
                    )
                    # Собираем строку вида 'typename or typename ...'
                    arg_types_name = " or ".join(
                        # Если '__qualname__' или '__name__' отсутствуют, класс типа возвращает сам себя
                        str(
                            getattr(arg_type, "__qualname__", False)
                            or getattr(arg_type, "__name__", arg_type)
                        )
                        for arg_type in arg_types
                    )
                    raise TypeError(f"Argument '{arg_name}' must be {arg_types_name}")
            # Проверка типов пройдена успешно. Вызываем оригинальную функцию
            return func(*args, **kwargs)

        return wrapper

    return decorate


# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
def abs_int(value: Any) -> int:
    """
    Проверяет значение на положительное целое.
    Если переданное значение невозможно представить как целое число,
    возбуждает исключение. Отрицательное число конвертирует в положительное.

    Args:
        value: Значение для проверки.

    Returns:
        int: Возвращает целое положительное число
    """
    try:
        result: int = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("The 'value' must support conversion to int.") from exc

    return (~result + 1) if result < 0 else result


# --------------------------------------------------------------------------------------
def get_ranges_index(
    list_len: int, range_len: int
) -> Generator[tuple[int, int], None, None]:
    """
    Функция-генератор, формирующая список индексов диапазонов заданной длины,
    на которые можно разбить исходный список длиной list_len.

    Args:
        list_len (int): Длина исходного списка.

        range_len (int): Размер диапазона.

    Yields:
        (start, end): Возвращает кортеж с начальным и конечным индексами диапазона.
    """
    # Корректируем возможные ошибки во входных параметрах
    _list_len: int = abs_int(list_len)
    _range_len: int = abs_int(range_len)
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
            yield (i, re if (re := (i + _range_len)) < _list_len else _list_len)


# --------------------------------------------------------------------------------------
def _is_srt(args: tuple[Iterator, bool]) -> bool:
    """
    Вспомогательная функция, поэлементно проверяющая отсортирован ли исходный список
    в зависимости от заданного направления сортировки. При первом ложном сравнении
    итерация прерывается.

    Args:
        args (tuple[Iterable, bool]): Кортеж параметров - итератор списка для проверки и направление сортировки

    Returns:
        bool: True/False - список отсортирован / не отсортирован.
    """
    elements, is_revers = args

    try:
        _prev = next(elements)
    except StopIteration:
        return True

    """
    Можно было использовать более компактный вариант:
    for _next in elements:
        if (_prev < _next) if is_revers else (_next < _prev):
            return False
        _prev = _next

    Но match работает немного быстрее
    """
    match is_revers:
        case True:
            for _next in elements:
                if _prev < _next:
                    return False
                _prev = _next
        case False:
            for _next in elements:
                if _next < _prev:
                    return False
                _prev = _next

    return True


# --------------------------------------------------------------------------------------
def is_sorted(
    elements: Sequence,
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

        revers (bool): Порядок сортировки. Defaults to False.

        rangesize (int | None): Размер диапазона, на который можно разбить список. Defaults to None.

    Returns:
        bool: True, если список отсортирован.
    """
    # Пустые списки или списки из одного элемента всегда отсортированы
    if (len_elements := len(elements)) < 2:
        return True

    cpu: int = multiprocessing.cpu_count()
    result: bool = True  # По умолчанию считаем список отсортированным

    # Если размер диапазона не задан, вычисляем исходя из производительности CPU
    if rangesize is None:
        range_size = cpu * max(round(len_elements**0.5), CPU_FREQUENCY)
    else:
        range_size = abs_int(rangesize)

    ranges_count: int = len_elements // range_size + int(
        bool(len_elements % range_size)
    )

    # Если исходный список можно разделить хотя бы на 2 подсписка
    # запускаем многозадачную обработку
    if ranges_count > 1:
        # Разбиваем исходный список на диапазоны и проверяем каждый диапазон в отдельном процессе.
        # Для каждого диапазона (кроме последнего) сравниваем последний элемент с первым элементом
        # следующего диапазона, для чего смещаем конечный индекс диапазона на 1.
        # Возможна ситуация, когда два отдельный подсписка отсортированы, но целый список нет
        # Например: [1,2,4,3,5,6]. Если разделить пополам, то оба подсписка будут отсортированы,
        # но при этом исходный полный список не отсортирован.
        margs_list = (
            (iter(elements[i_start : i_end + int(i_end < len_elements)]), revers)
            for i_start, i_end in get_ranges_index(len_elements, range_size)
        )

        # Запускаем пул параллельных процессов для проверки сортировки набора диапазонов
        # - результаты получаем сразу по готовности не дожидаясь завершения всех проверок
        # - возможно досрочное завершение обработки результатов
        with multiprocessing.Pool(processes=min(cpu, ranges_count)) as mpool:
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
    if m := abs_int(imonth) % 12:
        imonth = m
    else:
        imonth = 12

    iyear = abs_int(iyear)
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
        (abs_int(iday) + (13 * imonth - 1) // 5 + (5 * iyear - 7 * icentury) // 4 + 777)
        % 7
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
def is_include_elements(data: Iterable, pattern: Iterable) -> bool:
    """
    Проверяет вхождение одного набора данных в другой. Сортировка не требуется, порядок не важен,
    дублирование элементов допускается, относительный размер списков не принципиален.

    Args:
        data (Iterable[Any]): Список, с которым сравнивается pattern.

        pattern (Iterable[Any]): Проверяемый список на вхождение в data.

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
    # Иначе в проверяемом списке есть элементы, которых нет в исходном, либо их больше, чем в исходном
    return len(-cdata) == 0


# -------------------------------------------------------------------------------------------------
def ilen(iterable: Iterable) -> int:
    """
    Подсчитывает количество элементов в итераторе. Попытка достичь компромиса между
    скоростью и потреблением памяти.
    По сравнению с len(tuple(iterable)) работает медленнее, но при этом практически
    не потребляет память.

    Args:
        iterable (Iterable): Набор данных, который поддерживает итерации.

    Returns:
        int: Количество элементов в данных.
    """
    # Если объект данных поддерживает метод len, то используем встроенный механизм
    # if hasattr(iterable, "__len__"): альтернативный вариант
    if isinstance(iterable, Sized):
        return len(iterable)  # type: ignore
    # Бесконечный счетчик-итератор
    iter_counter = itertools.count()
    # Создаем очередь нулевой длины, которая используется только для наращивания
    # счетчика iter_counter и никаких данных не хранит.
    deque(zip(iterable, iter_counter), 0)
    # Возвращаем значение счетчика. Т.к. отсчет ведется с нуля, прибавляем единицу
    # return sum(1 for _ in iterable) # Альтернативный вариант
    return next(iter_counter)


# -------------------------------------------------------------------------------------------------
def is_int(val: Any) -> bool:
    try:
        _ = int(val)
    except Exception:
        return False
    else:
        return True


# -------------------------------------------------------------------------------------------------
class WarningToConsole:
    __slots__ = "__logger"

    def __init__(self, msg: str | None = None, logname: str | None = None) -> None:
        self.__logger: logging.Logger = logging.getLogger(
            __name__ if logname is None else logname
        )
        if self.__logger.getEffectiveLevel() > logging.WARNING:
            self.__logger.setLevel(logging.WARNING)
        # Формируем уникальное имя handler-ра для конкретного логгера
        handler_name: str = f"_{self.__logger.name}__{id(self.__logger)}"
        # Предотвращаем дублирование handler-ра
        if handler_name not in set(
            handler.name
            for handler in self.__logger.handlers
            if handler.name is not None
        ):
            formatter = logging.Formatter(
                "{asctime} [{levelname}] - {message}"
                if logname is None
                else "{asctime} [{levelname}] — {name} - {message}",
                datefmt="%d-%m-%Y %H:%M:%S",
                style="{",
            )
            handler = logging.StreamHandler(sys.stdout)
            handler.set_name(handler_name)
            handler.setLevel(logging.WARNING)
            handler.setFormatter(formatter)
            self.__logger.addHandler(handler)

        if msg is not None:
            self.warning(msg)

    def __call__(self, msg: str) -> None:
        self.warning(msg)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.__logger.warning(*args, **kwargs)


# -------------------------------------------------------------------------------------------------
def rinda_multiplication(a: int, b: int) -> int:
    """Выполняет перемножение двух целочисленных значений через сложение и битовые операций
    деления/умножения на 2. Учитывается знак множителей.

    Пошаговый алгоритм умножения (89 * 18):
    1. Определить минимальное значение (18).
    2. Пошагово 18 делим на 2 без остатка, а 89 умножаем на 2, посредством смещения на 1 бит вправо/влево.
    Деление/умножение выполняется попарно.
    3. При этом, четные значения, полученные после деления на 2, отбрасываем. Также отбрасываем
    соответствующую пару, полученную умножением на 2 второго значения.
    4. Когда деление на 2 достигнет единицы, цикл итераций деления/умножения прекращается.
    5. Оставшиеся значения, которые умножались на 2, суммируются. Полученная сумма есть результат перемножения.

    После второго шага получаем набор пар:
    (18, 89)
    (9, 178)
    (4, 356)
    (2, 712)
    (1, 1424)

    После третьего шага
    (9, 178)
    (1, 1424)

    На пятом шаге получаем результат: 178 + 1424 = 1602

    Args:
        a (int): Первый множитель
        b (int): Второй множитель

    Returns:
        int: Произведение
    """
    # Определяем знак результирующего произведения
    # Если одно из значений <0, то sign = (-1)**True = (-1)**1 = -1
    # Если оба <0 или >0, то sign = (-1)**False = (-1)**0 = 1
    sign: int = (-1) ** ((a < 0) ^ (b < 0))

    def _get_addendum(a, b) -> Generator[int, None, None]:
        """Внутренняя функция-генератор, которая выполняет попарное деление/умножение
        на 2, и фильтрует четные значения, полученные после деления на 2."""
        # Работаем с абсолютными целочисленными значениями
        # Для уменьшения количества итераций, выбираем наименьший множитель
        if (b := abs_int(b)) < (a := abs_int(a)):
            a, b = b, a

        while a > 0:
            if a & 1:
                yield b
            a >>= 1
            b <<= 1

    return sign * sum(_get_addendum(a, b))


# -------------------------------------------------------------------------------------------------
def inumber_to_digits(number: Any) -> list[int]:
    """Функция преобразования целого числа в список цифр.

    Example:
        1. inumber_to_digits(7362) -> [7, 3, 6, 2]
        2. inumber_to_digits(7362.7) -> [7, 3, 6, 2]
        3. inumber_to_digits('number') -> []

    Args:
        number (Any): Заданное целое число.

    Returns:
        list[int]: Список цифр.
    """
    try:
        number = int(number)
    except Exception:
        return []

    def get_digits(num: int) -> Generator[int, Any, None]:
        yield num % 10
        while num := num // 10:
            yield num % 10

    # Знак числа отбрасываем
    if number < 0:
        number = ~number + 1
    return list(get_digits(number))[::-1]


# -------------------------------------------------------------------------------------------------
def inumber_to_digits2(number: Any) -> tuple[int]:
    """Функция преобразования целого числа в список цифр.
    Альтернативный вариант. Используется итератор для генерации цифр.

    Args:
        number (Any): Заданное целое число.

    Returns:
        tuple[int]: Список цифр.
    """
    try:
        number = int(number)
    except Exception as exc:
        raise ValueError(f"impossible to represent {number} as an integer") from exc

    # Функция раскладывает число на цифры. Знак числа отбрасывается
    # Как только цифры закончатся, вернет None
    def get_digits():
        # Состояние между вызовами сохраняем в атрибуте 'num'
        # Инициализируем атрибут стартовым кортежем (number, 0)
        get_digits.num = getattr(get_digits, "num", (abs(number), 0))
        get_digits.num = divmod(get_digits.num[0], 10)
        return get_digits.num[1] if get_digits.num != (0, 0) else None

    # Итератор вызывает функцию до тех пор, пока не получит None
    return (*iter(get_digits, None),)[::-1]


# -------------------------------------------------------------------------------------------------


def unpack_fast(*args: object) -> Generator[object, Any, None]:
    """Распаковывает наборы данных (списки, словари, генераторы и т.п.) с любым уровнем
    вложенности в плоский список.

    Данная версия работает быстро, но потребляет память пропорционально количеству
    переданных данных. В очереди хранятся все распакованные объекты.

    Returns:
        Generator[object, Any, None]: Генерирует распакованные объекты.
    """
    # Объект _stop делит очередь пополам
    # До _stop уже обработанные распакованные объекты
    # После _stop объекты, требующие обработки и распаковки
    _stop = object()
    args_buff: deque[object] = deque()
    args_buff.append(_stop)
    # Загружаем в очередь все полученные функцией объекты
    args_buff.extend(args)
    while (arg := args_buff.pop()) != _stop:
        match arg:
            # Если это итерируемый объект (но не строка)
            # Распаковываем и помещаем в конец очереди после _stop
            case Iterable() if not isinstance(arg, (str, bytes)):
                args_buff.extend(arg)
            # Строки и не итерируемые объекты помещаем в очередь до _stop
            case _:
                args_buff.appendleft(arg)
    # В итоге в начале очереди получим упорядоченные распакованне объекты
    while args_buff:
        yield args_buff.popleft()


# -------------------------------------------------------------------------------------------------


def unpack_small(*args: object) -> Generator[object, Any, None]:
    """Распаковывает наборы данных (списки, словари, генераторы и т.п.) с любым уровнем
    вложенности в плоский список.

    Данная версия работает немного медленнее В очереди хранятся либо итераторы, либо
    распакованные объекты, которые необходимо отдать позже. Распакованные объекты
    хранятся частично, а итераторы имеют фиксированные размер.

    Returns:
        Generator[object, Any, None]: Генерирует распакованные объекты.
    """
    iters_buff: deque[object] = deque()
    # Т.к. args - это кортеж, делаем из него итератор и помещаем в очередь
    iters_buff.append(iter(args))
    while iters_buff:
        match obj := iters_buff.popleft():
            case Iterable() if not isinstance(obj, (str, bytes)):
                i = 0
                # Разбираем итерируемый объект
                for _obj in obj:
                    if isinstance(_obj, Iterable) and not isinstance(
                        _obj, (str, bytes)
                    ):
                        # Внутри может оказаться еще один итерируемый объект
                        iters_buff.insert(i, iter(_obj))
                        i += 1
                    elif i:  # Если i>0
                        # Иначе сохраняем не итерируемый объект на своем порядковом месте
                        # Когда до него дойдет очередь, отдадим в блоке 'case _:'
                        iters_buff.insert(i, _obj)
                        i += 1
                    else:  # Если i=0
                        # Если не итерируемый объект первый в списке, отдаем без сохранения
                        yield (_obj)
            case _:
                # Отдаем не итерируемый объект
                yield obj


# -------------------------------------------------------------------------------------------------


def unpack_least(*args: object) -> Generator[object, Any, None]:
    """Распаковывает наборы данных (списки, словари, генераторы и т.п.) с любым уровнем
    вложенности в плоский список.

    Самая медленная версия, но при этом потребление памяти минмимально. В очереди
    хранятся исключительно id обрабатываемых объектов - целочисленные значения.

    Returns:
        Generator[object, Any, None]: Генерирует распакованные объекты.
    """
    iters_buff: deque[int] = deque()
    # Работаем только с id объектов
    iters_buff.append(id(args))
    while iters_buff:
        # Получаем по id содержимое объекта
        match arg := ctypes.cast(iters_buff.popleft(), ctypes.py_object).value:
            # Если это итерируемый объект (но не строка)
            case Iterable() if not isinstance(arg, (str, bytes)):
                # Распаковываем объект, при этом индексируем
                i = 0
                for _arg in arg:
                    # Внутри может оказаться еще один итерируемый объект
                    # Либо сохраняем не итерируемый объект на своем порядковом месте
                    if (
                        isinstance(_arg, Iterable)
                        and not isinstance(_arg, (str, bytes))
                    ) or i > 0:
                        iters_buff.insert(i, id(_arg))
                        i += 1
                    else:  # Если i=0
                        # Если не итерируемый объект первый в списке, отдаем без сохранения
                        yield (_arg)
            case _:
                # Отдаем не итерируемый объект
                yield arg


# -------------------------------------------------------------------------------------------------


def get_qty_elements_cross(
    data: Iterable, qty: int, offset: int = 1
) -> Generator[Any, Any, None]:
    """Последовательно выбирает заданное количество значений из списка данных. При этом диапазона выбранных
    значений накладываются друг на друга. Например: (0, 1, 2) (1, 2, 3) (2, 3, 4) (3, 4, 5)...

    Example:
        1. get_qty_elements_cross([0, 1, 2, 3, 4, 5], 2) -> (0, 1) (1, 2) (2, 3) (3, 4) (4, 5)
        2. get_qty_elements_cross([0, 1, 2, 3, 4, 5], 2, 2) -> (0, 2) (1, 3) (2, 4) (3, 5)
        3. get_qty_elements_cross([0, 1, 2, 3, 4, 5], 2, 0) -> (0, 0) (1, 1) (2, 2) (3, 3) (4, 4) (5, 5)

    Args:
        data (Iterable): Список данных.
        qty (Iterable): Количество отбираемых значений.
        offset (Iterable): Смещение между значениями.

    Returns:
        Generator[Any, Any, None]: Последовательность кортежей с заданным количеством значений.
    """
    # Количество итераторов, равное количеству отбираемых значений, с заданным смещением
    iters = (itertools.islice(data, i * abs(offset), None) for i in range(qty))
    # Генерация кортежей с отобранными значениями
    yield from zip(*iters)


# -------------------------------------------------------------------------------------------------


def get_qty_elements_uncross(
    data: Iterable, qty: int, offset: int = 1
) -> Generator[Any, Any, None]:
    """Последовательно выбирает заданное количество значений из списка данных. При этом диапазона выбранных
    значений не пересекаются. Например: (0, 1, 2) (3, 4, 5) (6, 7, 8) (9, 10, 11)...

    Example:
        1. get_qty_elements_uncross([0, 1, 2, 3, 4, 5], 2) -> (0, 1) (2, 3) (4, 5)
        2. get_qty_elements_uncross([0, 1, 2, 3, 4, 5, 6], 2, 2) -> (0, 2) (4, 6)
        3. get_qty_elements_uncross([0, 1, 2, 3, 4, 5], 2, 0) -> (0, 0) (1, 1) (2, 2) (3, 3) (4, 4) (5, 5)

    Args:
        data (Iterable): Список данных.
        qty (Iterable): Количество отбираемых значений.
        offset (Iterable): Смещение между значениями.

    Returns:
        Generator[Any, Any, None]: Последовательность кортежей с заданным количеством значений.
    """
    # Базовый итератор с заданным смещением
    if offset != 0:
        data = itertools.islice(data, 0, None, abs(offset))
    # Количество итераторов по базовому итератору, равное количеству отбираемых значений
    iters = (itertools.islice(data, 0, None) for _ in range(qty))
    # Генерация кортежей с отобранными значениями
    yield from zip(*iters)


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
        f" is_includes_elements([1, 2, 3, 4, 5, 6, 7, 8, 8], [1, 2, 4, 8]) -> {is_include_elements([1, 2, 3, 4, 5, 6, 7, 8, 8], [1, 2, 4, 8])}"
    )

    print("\n- Перемножение двух целочисленных значений через сложение.")
    print(f" rinda_multiplication(89, -18) -> {rinda_multiplication(89, -18)}")

    print(
        "\n- Распаковывает наборы данных с любым уровнем вложенности в плоский список."
    )
    print(
        f" unpack_fast(0, 1, range(2,5), [(5, 6), 7, (8, 9)]) -> {list(unpack_fast(0, 1, range(2,5), [(5, 6), 7, (8, 9)]))}"
    )


# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = range(100_000_000)
    print(MiniTimers(is_sorted, data, timer="Best", repeat=10))

    main()
