import collections
import dataclasses
import functools
import inspect
import logging
import sys
import time
from collections.abc import Callable, Generator
from itertools import repeat
from typing import Any, Literal, Self, TypeAlias

import demo

TRepeat: TypeAlias = int | str | float
TLogger: TypeAlias = logging.Logger | str | None
TTimerType: TypeAlias = Literal["Timer", "Call", "Total", "Best", "Average"]


class TimersErrors(Exception):
    pass


class TimerCallError(TimersErrors):
    pass


class TimerStartError(TimersErrors):
    def __str__(self) -> str:
        return "Timer already started"


class TimerStopError(TimersErrors):
    def __str__(self) -> str:
        return "Timer not started"


class TimerNotStopError(TimersErrors):
    def __str__(self) -> str:
        return "Timer not stopped"


class TimerTypeError(TimersErrors):
    pass


@dataclasses.dataclass(slots=True)
class _TimersTime:
    """Private class for keeping elapsed time of Timers class."""

    # Последний сохраненный таймер
    _last_timer: TTimerType = "Timer"
    # Время счетчиков
    _timers_time: dict[TTimerType, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    # Текст логов для счетчиков
    _timers_log: dict[TTimerType, str] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(str)
    )

    def __str__(self) -> str:
        # Выводим лог последнего сохраненного таймера
        return self.log(self._last_timer)

    def reset(self, timer: TTimerType | None = None) -> None:
        if timer is None:
            # Сбрасываем все счетчики
            self._timers_time.clear()
            self._timers_log.clear()
        else:
            # Сбрасываем конкретный счетчик
            self._timers_time.pop(timer, 0.0)
            self._timers_log.pop(timer, "")

    def save(self, timer: TTimerType, time: float, log: str = "") -> None:
        self._timers_time[timer] = time
        self._timers_log[timer] = log
        self._last_timer = timer

    def time(self, timer: TTimerType | None = None) -> float:
        if timer is None:
            timer = self._last_timer
        return self._timers_time.get(timer, 0.0)

    def log(self, timer: TTimerType | None = None) -> str:
        if timer is None:
            timer = self._last_timer
        _str: str = self._timers_log.get(timer, f"{timer} time not measured!")
        # Если строка с текстом лога пуста, возвращаем просто время
        if len(_str) == 0:
            _str = f"{timer} time = {self.time(timer)}"
        return _str


class Timers:
    """Класс-таймер, для замеров времени выполнения кода.
    Поддерживает как ручной запуск/останов: start()/stop(), так и менеджер контекста 'with'
    и декорирование @Timer().register. Реализован функционал замера производительности вызываемого
    объекта с параметрами с указанием количество повторных запусков (по-умолчанию 1000).

    Args:
        time_source (Callable): Функция, используемая для измерения времени. Default: time.perf_counter
        is_accumulate (bool): Флаг, активирующий режим аккумулирования замеров. Default: False
        is_show (bool): Флаг, разрешающий вывод результатов замера на консоль. Default: True
        repeat (int): Количество повторных запусков вызываемого объекта. Default: 1000
        logger (Logger | str | None): Вывод результатов замеров. По-умолчанию на консоль. Default: None
    """

    def __init__(
        self,
        time_source: Callable = time.perf_counter,
        is_accumulate: bool = False,
        is_show: bool = True,
        repeat: TRepeat = 1000,
        # Позволяет перенаправить вывод тайминга. По-умолчанию вывод на консоль.
        logger: TLogger = None,
    ) -> None:
        self.time_source = time_source
        self.is_accumulate = is_accumulate
        self.is_show = is_show
        self.repeat = repeat
        # Если logger не задан, создаем дефолтовый для класса Timers
        self.logger = logger if logger is not None else type(self).__name__
        # Класс-хранилище всех измеряемых показателей времени
        self.__timers: _TimersTime = _TimersTime()
        # None - счетчик не запущен
        self.__start_time: float | None = None

    def __call__(
        self,
        func: Callable,
        *args: Any,
        timer: TTimerType = "Call",
        **kwargs: Any,
    ) -> Any:
        """Вычисляет время выполнения заданной функции с аргументами за N повторов.
        Измеренное время доступно через свойства time_xxxx. Неявно можно передать настроечные
        параметры time_source, repeat, is_accumulate, is_show и logger в виде именованных аргументов,
        иначе будут использованы значения соответствующих свойств экземпляра класса.
        Неявные настроечные параметры действуют единоразово только на момент вызова метода и
        не меняют параметры, заданные при инициализации экземпляра класса Timers.
        В режимах "Timer" и "Call" параметр repeat не используется.
        В режимах "Total", "Best" и "Average" параметр is_accumulate игнорируется.

        Examples:
            instance(func, *args, **kwargs, timer='Best', repeat=100)
            instance(func, *args, **kwargs, time_source=time.process_time, is_show=False)

        Args:
            func (Callable): Вызываемый объект для замера времени выполнения
            *args (Any): Список позиционных аргументов для вызываемого объекта
            **kwargs (Any): Список именованных аргументов для вызываемого объекта
            timer (str): Какое время нужно измерит. Total-общее, Best-лучшее, Average-среднее. Timer и Call - однократный вызов. Default: 'Call'
            time_source (Callable): Функция, используемая для измерения времени. Default: self.time_source
            repeat (int): Количество повторных запусков вызываемого объекта. Default: self.repeat
            is_accumulate (bool): Флаг, активирующий режим аккумулирования замеров. Default: self.is_accumulate
            is_show (bool): Флаг, разрешающий вывод результатов измерений на консоль. Default: self.is_show
            logger (Logger): Вывод результатов замеров в заданный поток. Default: self.logger

        Raises:
            RuntimeError: Если во время вызова исполняемого объекта произошла ошибка, генерируется исключение с
            указанием имени вызываемого объекта и его аргументов.

        Returns:
            Any: Результат, возвращаемый вызываемым объектом.
        """
        # Формируем список параметров, которые передаются в метод __init__.
        # Для каждого из полученных параметров выполняем три шага:
        # - пытаемся получить значение из входных параметров метода __call__ из kwargs
        # - иначе пытаемся получить значение из атрибута экземпляра класса Timers
        # - иначе подставляем значение по-умолчанию, заданное в методе __init__
        # Если значение по-умолчанию не задано, подставляется пустой object
        # В итоге будет сформирован словарь из имен параметров и их значений
        _attrs: dict[str, Any] = dict(
            (
                arg,
                kwargs.pop(
                    arg,
                    getattr(self, arg, False)
                    or getattr(self, "_" + arg, False)
                    or getattr(
                        self,
                        f"_{type(self).__name__}__{arg}",
                        val.default if val.default is not val.empty else object(),
                    ),
                ),
            )
            for arg, val in inspect.signature(self.__init__).parameters.items()
        )
        # Из полученного словаря формируем именованный кортеж для удобства
        _self = collections.namedtuple("SelfAttributes", _attrs.keys())(**_attrs)

        result: Any = None
        func_signature: str = f"{demo.get_object_modname(func)}({self._get_func_parameters(func, *args, **kwargs)})"
        _time: float = float("inf")

        try:
            match timer:
                case "Timer" | "Call":
                    _time = _self.time_source()
                    result = func(*args, **kwargs)
                    _time = _self.time_source() - _time
                    if _self.is_accumulate:
                        _time += self.__timers.time(timer)
                case "Total" | "Average":
                    _time = _self.time_source()
                    # itertools.repeat быстрее range
                    for _ in repeat(None, _self.repeat):
                        result = func(*args, **kwargs)
                    _time = _self.time_source() - _time
                    if timer == "Average":
                        _time = _time / _self.repeat
                case "Best":
                    for _ in repeat(None, _self.repeat):
                        _elapsed_time: float = _self.time_source()
                        result = func(*args, **kwargs)
                        _elapsed_time = _self.time_source() - _elapsed_time
                        if _elapsed_time < _time:
                            _time = _elapsed_time
                case _:
                    return func(*args, **kwargs)
        except Exception as exc:
            raise TimerCallError(f"Call error {func_signature}") from exc
        else:
            _log: str = f"{timer} time [{func_signature} -> {result}] = {_time}"
            self.__timers.save(timer, _time, _log)
            if _self.is_show:
                _self.logger.info(_log)

        return result

    def __repr__(self) -> str:
        # На случай, когда не удалось получить значение параметра
        _not_defined = object()
        _cls: str = type(self).__name__
        # Собираем из списка параметров метода __init__ строку инициализации экземпляра класса
        # с текущими значениями параметров, а не с переданными в момент инициализации (благодаря генератору)
        # Генератор формируем каждый раз заново, т.к. генератор одноразовый, а метод __repr__
        # может быть вызван многократно
        # Важно! Имя атрибута должно совпадать с именем параметра кроме начальных подчеркиваний
        get_init_parameters: Generator[tuple[str, Any], None, None] = (
            (
                attr,  # Параметр: somename
                # 'or' возвращает первый getattr, который успешен. Иначе _not_defined
                getattr(self, attr, False)  # Атрибут: self.somename
                or getattr(self, "_" + attr, False)  # Атрибут: self._somename
                # Атрибут: self.__somename
                or getattr(
                    self,
                    f"_{_cls}__{attr}",
                    val.default if val.default is not val.empty else _not_defined,
                ),
            )
            for attr, val in inspect.signature(self.__init__).parameters.items()
        )
        return "".join(
            (
                f"{_cls}(",
                ", ".join(
                    # Отбираем те параметры, которые удалось идентифицировать
                    f"{attr} = {demo.get_object_modname(value)}"
                    for attr, value in get_init_parameters
                    if value is not _not_defined
                ),
                ")",
            )
        )

    def __str__(self) -> str:
        # По-умолчанию выводим лог последнего сохраненного таймера
        return self.__timers.log()

    def log(self, timer: TTimerType | None = None) -> str:
        # Выводим лог заданного таймера или последнего сохраненного
        return self.__timers.log(timer)

    # Для поддержки протокола менеджера контекста, реализованы методы __enter__ и __exit__
    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        # Внутри контекстного менеджера возможен вызов self.reset()
        if self.__start_time is not None:
            self.stop()

    def register(self, func: Callable):
        """Позволяет использовать класс как декоратор: @Timers().register. При этом можно задать
        функцию источник времени и другие параметры для класса Timers.
        Допускается использовать один и тот же таймер и как декоратор, и как менеджер контекста и как ручной таймер.
        При этом декоратор может быть вложен в менеджер контента или в ручной таймер.

        Например:

        @Timers(time.process_time, is_accumulate=True, is_show=False, logger=MyLogger).register.
        Альтернатива, предварительно создать экземпляр класса Timers, настроить необходимые параметры и
        использовать экземпляр как декоратор: @instance.register.

        Параметр repeat для декоратора не используется и игнорируется.
        """

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return self(func, *args, **kwargs, timer="Call")

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
        try:
            _str: str = ", ".join(
                f"{arg_name}={arg_value!r}"
                for arg_name, arg_value in func_signature.bind(
                    *args, **dict(_kwargs)
                ).arguments.items()
            )
        except Exception:
            _str = ""
        return _str

    @staticmethod
    def _get_logger(logger_name: str) -> logging.Logger:
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
        # Формируем уникальное имя handler-ра для конкретного логгера
        handler_name: str = f"_{logger_name}__{id(logger)}"
        # Предотвращаем дублирование handler-ра
        if handler_name not in set(
            handler.name for handler in logger.handlers if handler.name is not None
        ):
            formatter = logging.Formatter(
                "{asctime} {name} [{levelname}]: {message}",
                datefmt="%d-%m-%Y %H:%M:%S",
                style="{",
            )
            # Перенаправляем весь вывод на консоль
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.set_name(handler_name)
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def start(self) -> None:
        """Запускает таймер."""
        # Если таймер уже был запущен
        if self.__start_time is not None:
            raise TimerStartError
        self.__start_time = self.time_source()

    def stop(self) -> float:
        """Останавливает таймер и вычисляет затраченное время. Если задан режим аккумулирования,
        вновь вычисленное время добавляется к накопленной ранее сумме. Это позволяет произвольно
        останавливать и возобновлять работу таймера без потери результата.

        Returns:
            float: Время работы таймера.
        """
        # Сразу же получаем текущее время, дабы минимизировать погрешность
        _time: float = self.time_source()

        # Если таймер еще не был запущен
        if self.__start_time is None:
            raise TimerStopError
        else:
            _time -= self.__start_time
            self.__start_time = None
            # Обрабатываем таймер. Аккумулируем, формируем лог и сохраняем
            _timer: TTimerType = "Timer"
            if self.is_accumulate:
                _time += self.__timers.time(_timer)
            _log: str = f"{_timer} running time = {_time}"
            self.__timers.save(_timer, _time, _log)

            if self.is_show:
                self.logger.info(_log)

        return self.time_timer

    def reset(self, timer: TTimerType | None = None) -> None:
        """Обнуляет заданный счетчик времени, либо все счетчики если timer=None.
        Для счетчика 'Timer' дополнительно останавливает таймер.
        """
        self.__timers.reset(timer)
        if timer == "Timer" or timer is None:
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
            raise TimerTypeError("Value 'time_source' must be Callable.")

    @property
    def is_accumulate(self) -> bool:
        """Флаг, активирующий режим аккумулирования измерений."""
        return self.__is_accumulate

    @is_accumulate.setter
    def is_accumulate(self, value: bool) -> None:
        try:
            self.__is_accumulate = bool(value)
        except (TypeError, ValueError) as exc:
            raise TimerTypeError("Value 'is_accumulate' must be bool.") from exc

    @property
    def is_show(self) -> bool:
        """Флаг, разрешающий вывод результатов замера на консоль."""
        return self.__is_show

    @is_show.setter
    def is_show(self, value: bool) -> None:
        try:
            self.__is_show = bool(value)
        except (TypeError, ValueError) as exc:
            raise TimerTypeError("Value 'is_show' must be bool.") from exc

    @property
    def repeat(self) -> int:
        """Количество повторных запусков вызываемого объекта."""
        return self.__repeat

    @repeat.setter
    def repeat(self, value: TRepeat) -> None:
        try:
            self.__repeat = int(value)
        except (TypeError, ValueError) as exc:
            raise TimerTypeError("Value 'repeat' must be int.") from exc

    @property
    def logger(self) -> logging.Logger:
        """Текущий логгер для вывода измерений времени."""
        return self.__logger

    @logger.setter
    def logger(self, logger: logging.Logger | str) -> None:
        match logger:
            case logging.Logger():
                # Меняем текущий логгер на новый или на самого себя с
                # измененными настройками
                self.__logger = logger
            case str():
                # Создаем новый логгер с заданным именем или перенастраиваем
                # существующий, добавляя отформатированный вывод на консоль.
                self.__logger = self._get_logger(logger)
            case _:
                raise TimerTypeError("Value 'logger' must be logging.Logger or str.")

    @property
    def is_running(self) -> bool:
        """Признак работающего таймера"""
        return self.__start_time is not None

    def time(self, timer: TTimerType | None = None) -> float:
        """Возвращает время заданного таймера или по-умолчанию последнего сохраненного"""
        if timer == "Timer" and self.is_running:
            raise TimerNotStopError
        return self.__timers.time(timer)

    @property
    def time_timer(self) -> float:
        """Время между start() и stop()"""
        return self.time("Timer")

    @property
    def time_call(self) -> float:
        """Время, измеренное декоратором"""
        return self.time("Call")

    @property
    def time_total(self) -> float:
        """Суммарное время выполнения вызываемого объекта за N повторов"""
        return self.time("Total")

    @property
    def time_best(self) -> float:
        """Лучшее время выполнения вызываемого объекта за N повторов"""
        return self.time("Best")

    @property
    def time_average(self) -> float:
        """Среднее время выполнения вызываемого объекта за N повторов"""
        return self.time("Average")


class MiniTimers:
    """Вычисляет время выполнения заданной функции с аргументами за N повторов (по-умолчанию N=1000).
    Мини версия класса Timers без ручного таймера, без декоратора, без менеджера контекста, без аккумулирования.
    Сохраняет результаты только последнего вызова.

    Args:
        func (Callable): Вызываемый объект для замера времени выполнения. Default: None
        *args (Any): Список позиционных аргументов для вызываемого объекта
        **kwds (Any): Список именованных аргументов для вызываемого объекта
        timer (str): Какое время нужно вычислить. Total-общее, Best-лучшее, Average-среднее. Timer и Call - однократный вызов. Default: 'Call'
        time_source (Callable): Функция, используемая для измерения времени. Default: time.perf_counter
        repeat (int): Количество повторных запусков вызываемого объекта. Default: 1000

    Raises:
        RuntimeError: Если во время вызова исполняемого объекта произошла ошибка, генерируется исключение с
        указанием имени вызываемого объекта и его аргументов.

    Returns:
        Any: Результат, возвращаемый вызываемым объектом.
    """

    def __init__(
        self,
        func: Callable | None = None,
        *args: Any,
        timer: TTimerType = "Call",
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
            self(func, *args, **kwds)

    def __call__(self, func: Callable, *args: Any, **kwds: Any) -> Any:
        self.__func_signature = f"{demo.get_object_modname(func)}({Timers._get_func_parameters(func, *args, **kwds)})"

        match self.timer:
            case "Timer" | "Call":
                _repeat: int = self.repeat
                try:
                    self.repeat = 1
                    self.__total_time(func, args, kwds)
                finally:
                    self.repeat = _repeat
            case "Total" | "Average":
                self.__total_time(func, args, kwds)
                if self.timer == "Average":
                    self.__time = self.__time / self.repeat
            case "Best":
                self.__best_time(func, args, kwds)
            case _:
                try:
                    return func(*args, **kwds)
                except Exception as exc:
                    raise RuntimeError(f"Call error {self.__func_signature}") from exc

        return self.result

    def __repr__(self) -> str:
        return "".join(
            (
                f"{type(self).__name__}(",
                "func, *args, **kwargs",
                f", timer={self.timer!r}",
                f", time_source={demo.get_object_modname(self.time_source)}",
                f", repeat={self.repeat!r}",
                ")",
            )
        )

    def __str__(self) -> str:
        return self.log

    def __total_time(self, func: Callable, args: Any, kwds: Any) -> None:
        result: Any = None
        try:
            start_time: float = self.time_source()
            for _ in repeat(None, self.repeat):
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
            for _ in repeat(None, self.repeat):
                _time: float = self.time_source()
                result = func(*args, **kwds)
                _time = self.time_source() - _time
                if _time < best_time:
                    best_time = _time
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
        return (
            f"{self.timer} time [{self.__func_signature} -> {self.result}] = {self.time}"
            if len(self.__func_signature) and self.time != float("inf")
            else f"{self.timer} time not measured!"
        )


if __name__ == "__main__":
    tmr = Timers()

    def countdown(n, t=0):
        while n > 0:
            n -= 1

    def listcomp(N):
        [х * 2 for х in range(N)]

    # print(MiniTimers(countdown, 50000, repeat=1000, timer="Best"))
    # tmr(listcomp, 1000000, repeat=10, timer="Best")
    # tmr(countdown, 50000, repeat=100, timer="Best")
    pass
