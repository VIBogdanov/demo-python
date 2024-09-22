import collections
import dataclasses
import functools
import inspect
import logging
import sys
import time
from typing import Any, Callable, Generator, Literal, Self, TypeAlias

TLogger: TypeAlias = logging.Logger | str | None
TTimerType: TypeAlias = Literal["Timer", "Call", "Total", "Best", "Average"]


@dataclasses.dataclass(slots=True)
class _TimeStorage:
    """Private class for keeping elapsed time of Timers class."""

    # Последний сохраненный таймер
    _last_timer: TTimerType = "Timer"
    # Время счетчиков
    timers_time: dict[TTimerType, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    # Текст логов для счетчиков
    timers_log: dict[TTimerType, str] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(str)
    )

    def __str__(self) -> str:
        # Выводим лог последнего сохраненного таймера
        return self.log(self._last_timer)

    def reset(self, timer: TTimerType | None = None) -> None:
        if timer is None:
            # Сбрасываем все счетчики
            self.timers_time.clear()
            self.timers_log.clear()
        else:
            # Сбрасываем конкретный счетчик
            self.timers_time.pop(timer, 0.0)
            self.timers_log.pop(timer, "")

    def save(self, timer: TTimerType, time: float, log: str = "") -> None:
        self.timers_time[timer] = time
        self.timers_log[timer] = log
        self._last_timer = timer

    def time(self, timer: TTimerType) -> float:
        return self.timers_time.get(timer, 0.0)

    def log(self, timer: TTimerType) -> str:
        _str: str = self.timers_log.get(timer, f"{timer} time not calculated!")
        # Если строка с текстом лога пуста, возвращаем просто время
        if len(_str) == 0:
            _str = str(self.time(timer))
        return _str


class Timers:
    """Класс-таймер, для замеров времени выполнения кода.
    Поддерживает как ручной запуск/останов: start()/stop(), так и менеджер контента 'with'
    и декорирование @Timer(). Реализован функционал замера производительности вызываемого
    объекта с параметрами с указанием количество повторных запусков (по-умолчанию 1000).

    Args:
        time_source (Callable): Функция, используемая для измерения времени. Default: time.perf_counter
        is_accumulate (bool): Флаг, активирующий режим аккумулирования замеров. Default: False
        is_show (bool): Флаг, разрешающий вывод результатов замера на консоль. Default: True
        repeat (int): Количество повторных запусков вызываемого объекта. Default: 1000
        logger (Logger | str | None): Вывод результатов замеров. По-умолчанию на консоль. Default: None
    """

    __slots__ = (
        "__time_source",
        "__is_accumulate",
        "__is_show",
        "__repeat",
        "__timers",
        "__start_time",
        "__logger",
        "__dict__",
        "__weakref__",
    )

    def __init__(
        self,
        time_source: Callable = time.perf_counter,
        is_accumulate: bool = False,
        is_show: bool = True,
        repeat: int = 1000,
        # Позволяет перенаправить вывод тайминга. По-умолчанию вывод на консоль.
        logger: TLogger = None,
    ) -> None:
        self.__time_source: Callable = time_source
        self.__is_accumulate: bool = is_accumulate
        self.__is_show: bool = is_show
        self.__repeat: int = repeat
        # Класс-хранилище всех измеряемых показателей времени
        self.__timers: _TimeStorage = _TimeStorage()
        # None - счетчик не запущен
        self.__start_time: float | None = None

        match logger:
            # Когда передан готовый объект Logger
            case logging.Logger():
                self.__logger: logging.Logger = logger
            # Когда предано имя логгера, который либо уже существует,
            # либо требуется создать логгер с заданным именем
            case str():
                self.__logger = self._get_loginfo(logger)
            # Иначе создаем свой собственный логгер
            case _:
                self.__logger = self._get_loginfo(self.__class__.__name__)

    def __call__(
        self,
        func: Callable,
        *args: Any,
        timer: TTimerType = "Total",
        **kwds: Any,
    ) -> Any:
        """Вычисляет время выполнения заданной функции с аргументами за N повторов.
        Измеренное время доступно через свойства time_xxxx. Неявно можно передать настроечные
        параметры time_source, repeat, is_accumulate и is_show в виде именованных аргументов,
        иначе будут использованы значения соответствующих свойств экземпляра класса.
        Неявные настроечные параметры действуют единоразово только на момент вызова метода и
        не меняют параметры, заданные при инициализации экземпляра класса Timers.

        Например: instance(func, *args, **kwds, repeat=100, is_accumulate=False)

        Args:
            func (Callable): Вызываемый объект для замера времени выполнения
            *args (Any): Список позиционных аргументов для вызываемого объекта
            **kwds (Any): Список именованных аргументов для вызываемого объекта
            timer (str): Какое время нужно вычислить. Total-общее, Best-лучшее, Average-среднее. Default: 'Total'
            time_source (Callable): Функция, используемая для измерения времени. Default: self.time_source
            repeat (int): Количество повторных запусков вызываемого объекта. Default: self.repeat
            is_accumulate (bool): Флаг, активирующий режим аккумулирования замеров. Default: self.is_accumulate
            is_show (bool): Флаг, разрешающий вывод результатов измерений на консоль. Default: self.is_show

        Raises:
            RuntimeError: Если во время вызова исполняемого объекта произошла ошибка, генерируется исключение с
            указанием имени вызываемого объекта и его аргументов.

        Returns:
            Any: Результат, возвращаемый вызываемым объектом.
        """
        # Извлекаем из входных аргументов настроечные параметры для time_source, _is_accumulate, is_show и repeat,
        # если они были заданы, иначе используем настройки текущего экземпляра класса Timers
        _time_source: Callable = kwds.pop("time_source", self.time_source)
        _is_accumulate: bool = kwds.pop("is_accumulate", self.is_accumulate)
        _is_show: bool = kwds.pop("is_show", self.is_show)
        _repeat: int = kwds.pop("repeat", self.repeat)

        result: Any = None
        func_signature: str = f"{func.__module__}.{func.__name__}({self._get_func_parameters(func, *args, **kwds)})"
        _time: float = float("inf")

        try:
            match timer:
                case "Timer" | "Call":
                    start_time: float = _time_source()
                    result = func(*args, **kwds)
                    _time = _time_source() - start_time
                    if _is_accumulate:
                        _time += self.__timers.time(timer)
                case "Total" | "Average":
                    start_time: float = _time_source()
                    for _ in range(_repeat):
                        result = func(*args, **kwds)
                    _time = _time_source() - start_time
                    if timer == "Average":
                        _time = _time / _repeat
                case "Best":
                    for _ in range(_repeat):
                        start_time: float = _time_source()
                        result = func(*args, **kwds)
                        elapsed_time: float = _time_source() - start_time
                        if elapsed_time < _time:
                            _time = elapsed_time
                case _:
                    return func(*args, **kwds)
        except Exception as exc:
            raise RuntimeError(f"Call error {func_signature}") from exc
        else:
            _log: str = f"{timer} time [{func_signature} -> {result}] = {_time}"
            self.__timers.save(timer, _time, _log)
            if _is_show:
                self.__logger.info(_log)

        return result

    def __repr__(self) -> str:
        return "".join(
            (
                f"{self.__class__.__name__}",
                f"(time_source={self.time_source.__module__}.{self.time_source.__name__}",
                f", is_accumulate={self.is_accumulate!r}",
                f", is_show={self.is_show!r}",
                f", repeat={self.repeat!r}",
                f", logger={self.__logger!r}",
                ")",
            )
        )

    def __str__(self) -> str:
        # Выводим __str__ класса _TimeStorage - лог последнего сохраненного таймера
        return str(self.__timers)

    def log(self, timer: TTimerType) -> str:
        # Выводим лог конкретного таймера, если он был ранее сохранен
        return self.__timers.log(timer)

    # Для поддержки протокола менеджера контента, реализованы методы __enter__ и __exit__
    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        # Внутри контентного менеджера возможен вызов self.reset()
        if self.__start_time is not None:
            self.stop()

    def wrapper(self, func: Callable):
        """Позволяет использовать класс как декоратор: @Timers().wrapper. При этом можно задать
        функцию источник времени и другие параметры для класса Timers.

        Например: @Timers(time.process_time, is_accumulate=True, is_show=False, logger=MyLogger).wrapper.
        Альтернатива, предварительно создать экземпляр класса Timers, настроить необходимые параметры и
        использовать экземпляр как декоратор: @instance.wrapper.
        Параметр repeat для декоратора не используется и игнорируется.
        """

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            result: Any = None
            func_signature: str = f"{func.__module__}.{func.__name__}({self._get_func_parameters(func, *args, **kwargs)})"
            # Позволяет использовать один и тот же таймер и как декоратор, и как менеджер контента и как ручной таймер.
            # При этом декоратор может быть вложен в менеджер контента или в ручной таймер.
            try:
                start_time: float = self.time_source()
                result = func(*args, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Call error {func_signature}") from exc
            else:
                _time: float = self.time_source() - start_time
                if self.is_accumulate:
                    _time += self.__timers.time("Call")
                _log: str = f"Call time [{func_signature} -> {result}] = {_time}"
                self.__timers.save("Call", _time, _log)

                if self.is_show:
                    self.__logger.info(_log)

            return result

        return _wrapper

    @staticmethod
    def _get_func_parameters(func: Callable, *args: Any, **kwargs: Any) -> str:
        """Формирует строку со списком аргументов и их значений вызываемого объекта,
        который передается классу Timers() для замера времени выполнения.

        Args:
            func (Callable): Вызываемый объект.
            *args (Any): Позиционные аргументы.
            **kwargs (Any): Именованные аргументы.

        Returns:
            str: Строка вида 'аргумент = значение, ...'
        """
        func_signature: inspect.Signature = inspect.signature(func)
        # Переупаковываем словарь именованных аргументов, оставляя только те, что заданы
        # в сигнатуре вызываемого объекта, т.к. дополнительно в списке именованных
        # аргументов могут присутствовать аргументы специфичные для класса Timers()
        _kwargs: Generator[tuple[str, Any], None, None] = (
            (k, v) for k, v in kwargs.items() if k in func_signature.parameters.keys()
        )
        # Формируем строку аргументов и их значений, переданных в вызываемый объект
        _str: str = ""
        try:
            _str: str = ", ".join(
                f"{arg_name}={arg_value}"
                for arg_name, arg_value in func_signature.bind(
                    *args, **dict(_kwargs)
                ).arguments.items()
            )
        except Exception:
            pass
        return _str

    @staticmethod
    def _get_loginfo(logger_name: str) -> logging.Logger:
        """Создает логгер для класса Timers(), который осуществляет
        отформатированный вывод на консоль.

        Args:
            logger_name (str): Имя логгера.

        Returns:
            TLogger: Объект логгера.
        """
        logger: logging.Logger = logging.getLogger(logger_name)
        # Если текущего уровня надостаточно для сообщений типа INFO
        if logger.getEffectiveLevel() > logging.INFO:
            logger.setLevel(logging.INFO)
        # Перенаправляем весь вывод на консоль
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            "{asctime} {name} [{levelname}]: {message}",
            datefmt="%d-%m-%Y %H:%M:%S",
            style="{",
        )
        handler.setLevel(logging.INFO)  # Необязательно
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def start(self) -> None:
        """Запускает таймер."""
        if self.__start_time is not None:
            raise RuntimeError("Timer already started")
        self.__start_time = self.time_source()

    def stop(self) -> float:
        """Останавливает таймер и вычисляет затраченное время. Если задан режим аккумулирования,
        вновь вычисленное время добавляется к накопленной ранее сумме. Это позволяет произвольно
        останавливать и возобновлять работу таймера без потери результата.

        Returns:
            float: Время работы таймера.
        """
        if self.__start_time is None:
            raise RuntimeError("Timer not started")

        _time: float = self.time_source() - self.__start_time
        if self.is_accumulate:
            _time += self.__timers.time("Timer")
        _log: str = f"Timer running time = {_time}"
        self.__timers.save("Timer", _time, _log)
        self.__start_time = None

        if self.is_show:
            self.__logger.info(_log)

        return self.time_timer

    def reset(self) -> None:
        """Обнуляет все счетчики времени и останавливает таймер,
        если он был запущен в момент вызова reset.
        """
        self.__timers.reset()
        self.__start_time = None

    def restart(self) -> None:
        """Сбрасывает счетчик ручного таймера start/stop и
        запускает таймер заново.
        """
        self.__timers.reset("Timer")
        self.__start_time = self.time_source()

    @property
    def time_source(self) -> Callable:
        """Функция, используемая для измерения времени."""
        return self.__time_source

    @time_source.setter
    def time_source(self, func: Callable) -> None:
        if isinstance(func, Callable):
            self.__time_source = func
        else:
            raise RuntimeError("Value must be Callable.")

    @property
    def is_accumulate(self) -> bool:
        """Флаг, активирующий режим аккумулирования измерений."""
        return self.__is_accumulate

    @is_accumulate.setter
    def is_accumulate(self, value: bool) -> None:
        try:
            self.__is_accumulate = bool(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Value must be bool.") from exc

    @property
    def is_show(self) -> bool:
        """Флаг, разрешающий вывод результатов замера на консоль."""
        return self.__is_show

    @is_show.setter
    def is_show(self, value: bool) -> None:
        try:
            self.__is_show = bool(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Value must be bool.") from exc

    @property
    def repeat(self) -> int:
        """Количество повторных запусков вызываемого объекта."""
        return self.__repeat

    @repeat.setter
    def repeat(self, value: int) -> None:
        try:
            self.__repeat = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Value must be bool.") from exc

    @property
    def is_running(self) -> bool:
        """Признак работающего таймера"""
        return self.__start_time is not None

    @property
    def time_timer(self) -> float:
        """Время между start() и stop()"""
        if self.__start_time is not None:
            raise RuntimeError("Timer not stopped")
        return self.__timers.time("Timer")

    @property
    def time_call(self) -> float:
        """Время, измеренное декоратором"""
        return self.__timers.time("Call")

    @property
    def time_total(self) -> float:
        """Суммарное время выполнения вызываемого объекта за N повторов"""
        return self.__timers.time("Total")

    @property
    def time_best(self) -> float:
        """Лучшее время выполнения вызываемого объекта за N повторов"""
        return self.__timers.time("Best")

    @property
    def time_average(self) -> float:
        """Среднее время выполнения вызываемого объекта за N повторов"""
        return self.__timers.time("Average")


class MiniTimers:
    """Вычисляет время выполнения заданной функции с аргументами за N повторов (по-умолчанию N=1000).
    Мини версия класса Timers без ручного таймера, без декоратора, без менеджера контента, без аккумулирования.
    Сохраняет результаты только последнего вызова.

    Args:
        func (Callable): Вызываемый объект для замера времени выполнения. Default: None
        *args (Any): Список позиционных аргументов для вызываемого объекта
        **kwds (Any): Список именованных аргументов для вызываемого объекта
        timer (str): Какое время нужно вычислить. Total-общее, Best-лучшее, Average-среднее. Default: 'Total'
        time_source (Callable): Функция, используемая для измерения времени. Default: time.perf_counter
        repeat (int): Количество повторных запусков вызываемого объекта. Default: 1000

    Raises:
        RuntimeError: Если во время вызова исполняемого объекта произошла ошибка, генерируется исключение с
        указанием имени вызываемого объекта и его аргументов.
    """

    __slots__ = (
        "time_source",
        "repeat",
        "timer",
        "__result",
        "__time",
        "__func_signature",
    )

    def __init__(
        self,
        func: Callable | None = None,
        *args: Any,
        timer: TTimerType = "Total",
        time_source: Callable = time.perf_counter,
        repeat: int = 1000,
        **kwds: Any,
    ) -> None:
        self.timer: TTimerType = timer
        self.time_source: Callable = time_source
        self.repeat: int = repeat
        self.__result: Any = None
        self.__time: float = float("inf")
        self.__func_signature: str = ""
        if func is not None and isinstance(func, Callable):
            self.__call__(func, *args, **kwds)

    def __call__(self, func: Callable, *args: Any, **kwds: Any) -> Any:
        self.__func_signature = f"{func.__module__}.{func.__name__}({Timers._get_func_parameters(func, *args, **kwds)})"

        match self.timer:
            case "Timer" | "Call":
                _repeat: int = self.repeat
                self.repeat = 1
                self.__total_time(func, args, kwds)
                self.repeat = _repeat
            case "Total":
                self.__total_time(func, args, kwds)
            case "Best":
                self.__best_time(func, args, kwds)
            case "Average":
                self.__total_time(func, args, kwds)
                self.__time = self.__time / self.repeat

        return self.result

    def __repr__(self) -> str:
        return "".join(
            (
                f"{self.__class__.__name__}",
                "(func, *args, **kwds",
                f", (timer={self.timer!r}",
                f", (time_source={self.time_source!r}",
                f", repeat={self.repeat!r}",
                ")",
            )
        )

    def __str__(self) -> str:
        return (
            f"{self.timer} time [{self.__func_signature} -> {self.result}] = {self.time}"
            if len(self.__func_signature) and self.time != float("inf")
            else ""
        )

    def __total_time(self, func: Callable, args: Any, kwds: Any) -> None:
        result: Any = None
        try:
            start_time: float = self.time_source()
            for _ in range(self.repeat):
                result = func(*args, **kwds)
        except Exception as exc:
            raise RuntimeError(f"Call error {self.__func_signature}") from exc
        else:
            # Обновляем через промежуточные переменные, дабы защититься от сбоя в блоке try и
            # не остаться в неопределенном состоянии. Обновляем только в случае полного успеха.
            self.__time = self.time_source() - start_time
            self.__result = result

    def __best_time(self, func: Callable, args: Any, kwds: Any) -> None:
        result: Any = None
        best_time: float = float("inf")
        try:
            for _ in range(self.repeat):
                start_time: float = self.time_source()
                result = func(*args, **kwds)
                elapsed_time: float = self.time_source() - start_time
                if elapsed_time < best_time:
                    best_time = elapsed_time
        except Exception as exc:
            raise RuntimeError(f"Call error {self.__func_signature}") from exc
        else:
            # Обновляем через промежуточные переменные, дабы защититься от сбоя в блоке try и
            # не остаться в неопределенном состоянии. Обновляем только в случае полного успеха.
            self.__time = best_time
            self.__result = result

    @property
    def result(self) -> Any:
        return self.__result

    @property
    def time(self) -> float:
        return self.__time

    @property
    def log(self) -> str:
        return self.__str__()
