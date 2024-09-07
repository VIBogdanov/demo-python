from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from functools import wraps
from inspect import Signature, signature
from itertools import count
from logging import Formatter, Logger, StreamHandler, getLogger
from multiprocessing import Pool, cpu_count
from time import perf_counter
from typing import Any, Self

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
        def samefunction(x, y, z=4.5):
            pass
        # Альтернативный вариант
        @type_checking(y = (int, str), x = int)
        def samefunction(x, y, z=4.5):
            pass
        # Результат работы декоратора
        samefunction(1, 3, z=123.5)   #OK
        samefunction(1, '3', z=123.5)   #OK
        samefunction(1, 3)   #OK
        samefunction('1', 3, z=123.5)   #Error
        samefunction(1, 3, z=123)   #Error for first variant
        samefunction(1, 3.4)   #Error
    """

    def decorate(func):
        # В режиме оптимизации отключаем декоратор
        if not __debug__:
            return func
        # Формируем словарь, связывающий арнументы функции с требованиями типов, заданными в декораторе
        func_signature: Signature = signature(func)
        args_types: OrderedDict[str, Any] = func_signature.bind_partial(
            *type_args, **type_kwargs
        ).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Формируем словарь с именами преданных функции аргументов и их значениями
            for arg_name, arg_value in func_signature.bind(
                *args, **kwargs
            ).arguments.items():
                # Если для данного аргумента задана проверка типа
                if arg_name in args_types:
                    # Если тип значения аргумента не соответствует заданному в декораторе
                    if not isinstance(arg_value, args_types[arg_name]):
                        raise TypeError(
                            f"Argument '{arg_name}' must be {args_types[arg_name]}"
                        )
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
        raise RuntimeError("The value must support conversion to int.") from exc

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

    cpu: int = cpu_count()
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
def is_includes_elements(data: Iterable, pattern: Iterable) -> bool:
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
    if hasattr(iterable, "__len__"):
        return len(iterable)  # type: ignore
    # Бесконечный счетчик-итератор
    iter_counter = count()
    # Создаем очередь нулевой длины, которая используется только для наращивания
    # счетчика iter_counter, и никаких данных не хранит.
    deque(zip(iterable, iter_counter), 0)
    # Возвращаем значение счетчика. Т.к. отсчет ведется с нуля, прибавляем единицу
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
class Timer:
    """Класс-таймер, для замеров времени выполнения кода.
    Поддерживает как ручной запуск/останов: start()/stop(), так и менеджер контента 'with'
    и декорирование @Timer().

    При инициализации или в процессе использования задаются функция источник времени и
    признак накопления измеренных интервалов. По умолчанию используется time.perf_counter
    и накопление отключено.
    """

    __slots__ = (
        "__elapsed_time",
        "__time_source",
        "__start_time",
        "__accumulate",
        "__dict__",
        "__weakref__",
    )

    def __init__(
        self, time_source: Callable = perf_counter, is_accumulate: bool = False
    ) -> None:
        self.__elapsed_time: float = 0.0
        self.__time_source = time_source
        self.__start_time = None
        self.__accumulate: bool = is_accumulate

    # Реализация метода __call__ в виде декоратора позволяет использовать класс как декоратор: @Timer()
    # При этом можно задать функцию источник времени. Например: @Timer(time.process_time)
    # Возможно задать и параметр is_accumulate, но он будет проигнорирован
    def __call__(self, func: Callable):
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            result: Any = None
            func_parameters: str = ", ".join(
                f"{arg_name}={arg_value}"
                for arg_name, arg_value in signature(func)
                .bind(*args, **kwargs)
                .arguments.items()
            )
            # Для декоратора используем свои собственные временные метки start и elapsed.
            # Это позволяет использовать один и тот же таймер и как декоратор, и как менеджер контента.
            # При этом декоратор может быть вложен в менеджер контента или в ручной таймер.
            start_time = self.__time_source()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Call error {func.__module__}.{func.__name__}({func_parameters})"
                ) from exc
            else:
                # При измерении интервала времени для декоратора аккумулирование не используется.
                # Для декоратора накопление не имеет смысла, т.к. измеряется конкретная функция
                # в конкретный момент ее вызова
                # Тонкий момент: можно было бы встроить вычисление elapsed_time прямо в строку форматирования
                # оператора print, но тогда в измерение интервала времени было бы внесено искажение,
                # связанное c затратами на формирование отформатированной строки для оператора print
                elapsed_time = self.__time_source() - start_time
                print(
                    f"{func.__module__}.{func.__name__}({func_parameters}) : {elapsed_time}"
                )
            return result

        return _wrapper

    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        return f"{_cls}(time_source={self.__time_source!r}, is_accumulate={self.__accumulate!r})"

    def start(self) -> None:
        if self.__start_time is not None:
            raise RuntimeError("Timer already started")
        self.__start_time = self.__time_source()

    def stop(self) -> float:
        if self.__start_time is None:
            raise RuntimeError("Timer not started")
        if self.__accumulate:
            self.__elapsed_time += self.__time_source() - self.__start_time
        else:
            self.__elapsed_time = self.__time_source() - self.__start_time
        self.__start_time = None
        return self.__elapsed_time

    def reset(self) -> None:
        self.__elapsed_time = 0.0
        self.__start_time = None

    @property
    def time_sourse(self):
        return self.__time_source

    @time_sourse.setter
    def time_sourse(self, func: Callable) -> None:
        self.__time_source = func

    @property
    def accumulate(self) -> bool:
        return self.__accumulate

    @accumulate.setter
    def accumulate(self, value: bool) -> None:
        try:
            self.__accumulate = bool(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Accumulate must be bool.") from exc

    # Признак работающего таймера
    @property
    def is_running(self) -> bool:
        return self.__start_time is not None

    # Измеренное время между start() и stop()
    @property
    def elapsed(self) -> float:
        if self.__start_time is not None:
            raise RuntimeError("Timer not stopped")
        return self.__elapsed_time

    # Для поддержки протокола менеджера контента, реализованы методы __enter__ и __exit__
    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# -------------------------------------------------------------------------------------------------
class WarningToConsole:
    __slots__ = "__logger"

    def __init__(self, msg: str | None = None, logname: str | None = None) -> None:
        self.__logger: Logger = getLogger(__name__ if logname is None else logname)
        _handler = StreamHandler()
        _formatter = Formatter(
            "{asctime} [{levelname}] - {message}"
            if logname is None
            else "{asctime} [{levelname}] — {name} - {message}",
            datefmt="%d-%m-%Y %H:%M:%S",
            style="{",
        )
        _handler.setFormatter(_formatter)
        self.__logger.addHandler(_handler)
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

    # Внутренняя функция-генератор, которая выполняет попарное деление/умножение
    # на 2, и фильтрует четные значения, полученные после деления на 2.
    def _get_addendum(a, b) -> Generator[int, None, None]:
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

    print("\n- Перемножение двух целочисленных значений через сложение.")
    print(f" rinda_multiplication(89, -18) -> {rinda_multiplication(89, -18)}")


# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = range(100_000_000)

    with Timer() as tm:
        res = is_sorted(data)
    print(f"Общее время выполнения is_sorted({res}):", tm.elapsed)

    main()
