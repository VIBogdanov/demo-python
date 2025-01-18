import collections
import dataclasses
import functools
import inspect
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from itertools import repeat
from typing import Any, Literal, Self, TypeAlias, cast, get_args

import demo

TRepeat: TypeAlias = int | str | float
TLogger: TypeAlias = logging.Logger | str | None
TTimerType: TypeAlias = Literal["Timer", "Call", "Total", "Best", "Average"]
TAttrMode: TypeAlias = Literal["Public", "Private", "Protected"]


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


class _TimersTypes(ABC):
    def __init__(
        self, attr_name: str = "", attr_mode: TAttrMode = "Public", **kwargs
    ) -> None:
        """Парметр attr_mode означает следующее:
        - Public - имя атрибута устанавливается равным attr_name без изменений
        - Private - к имени атрибута добавляется префикс '_'
        - Protected - к имени атрибута добавляется префикс '_ClassName__'

        Все действия по формированию имени атрибута выполняются в методе __set_name__.
        Если в методе __init__ аргумент attr_name не задан (по-умолчанию), в __set_name__
        он устанавливается автоматически равным имени атрибута управляемого класса.
        """
        super().__init__()
        # Можно задать собственное имя атрибута, если требуется
        # Очищаем имя атрибута от начальных и конечных пробелов
        self._attr_name: str = attr_name.strip()
        self._attr_mode: TAttrMode = attr_mode
        # Сохраняем любые переданные именованные аргументы
        # В данной реализации этот функционал не используется
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set_name__(self, cls_owner, attr_name: str) -> None:
        self._set_name = attr_name
        # Если при инициализации имя не задано, устанавливаем имя атрибута управляемого класса
        _name = self._attr_name if self._attr_name else self._set_name
        match self._attr_mode:
            case "Public":
                self._attr_name = _name
            case "Private":
                self._attr_name = f"_{_name}"
            case "Protected":
                self._attr_name = f"_{cls_owner.__name__}__{_name}"

    def __set__(self, instance, value: Any) -> None:
        value = self.type_check(self._attr_name, value)
        if self._attr_name == self._set_name:
            # Если имя атрибута управляемого класса и имя атрибута экземпляра совпадают,
            # то setattr приведет к зацикливанию, потому сохраняем через словарь
            instance.__dict__[self._attr_name] = value
        else:
            setattr(instance, self._attr_name, value)

    def __get__(self, instance, cls_owner=None) -> Any:
        # Если атрибут вызван из класса, возвращаем класс-дескриптор
        if instance is None:
            return self
        if self._attr_name == self._set_name:
            return instance.__dict__[self._attr_name]
        else:
            return getattr(instance, self._attr_name)

    @abstractmethod
    def type_check(self, attr_name: str, value: Any) -> Any:
        """Проверить и вернуть значение или возбудить исключение TimerTypeError"""
        return value


class TypeChecker(_TimersTypes):
    expected_type: Any = None

    def type_check(self, attr_name: str, value: Any) -> Any:
        self.expected_type = (
            type(None) if self.expected_type is None else self.expected_type
        )
        # Тип можно задавать как, например: int или int() или 12
        if not isinstance(value, (self.expected_type, type(self.expected_type))):
            raise TimerTypeError(
                f"Value for '{attr_name}' must be {self.expected_type}."
            )
        # Вызываем super() для отработки множественного наследования
        return super().type_check(attr_name, value)


class PositiveValue(_TimersTypes):
    def type_check(self, attr_name: str, value: Any) -> Any:
        try:
            if value <= 0:
                raise TimerTypeError(
                    f"Value for '{attr_name}' must be greater than zero."
                )
        except TypeError as exc:
            raise TimerTypeError(
                f"Value for '{attr_name}' must support '<=' operator."
            ) from exc
        return super().type_check(attr_name, value)


class TypeBool(TypeChecker):
    expected_type = bool


class TypeInteger(TypeChecker):
    expected_type = int


class TypePosInteger(TypeInteger, PositiveValue):
    pass


class TypeCallable(TypeChecker):
    expected_type = Callable


class TypeLogger(_TimersTypes):
    def __set__(self, instance, value: TLogger) -> None:
        # Если logger не задан, создаем дефолтовый для класса Timers
        super().__set__(
            instance, value if value is not None else type(instance).__name__
        )

    def __get__(self, instance, owner=None):
        try:
            return super().__get__(instance, owner)
        except Exception:
            # Если логгер еще не был установлен, возвращаем дефолтовый
            return self._get_logger(type(instance).__name__)

    def type_check(self, attr_name: str, value: Any) -> logging.Logger:
        match value:
            case logging.Logger():
                # Меняем текущий логгер на новый или на самого себя с
                # измененными настройками
                pass
            case str():
                # Создаем новый логгер с заданным именем или перенастраиваем
                # существующий, добавляя отформатированный вывод на консоль.
                value = self._get_logger(value)
            case _:
                raise TimerTypeError(
                    f"Value for '{attr_name}' must be logging.Logger or str."
                )
        return super().type_check(attr_name, value)

    @staticmethod
    def _get_logger(logger_name: str) -> logging.Logger:
        """Создает логгер для класса Timers(), который осуществляет
        отформатированный вывод на консоль.

        Args:
            logger_name (str): Имя логгера.

        Returns:
            Logger: Объект логгера.
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
        """Удаляет сохраненные время и лог для заданного таймера.
        Если конкретный таймер не задан (None), удаляет все таймеры."""
        if timer is None:
            # Сбрасываем все счетчики
            self._timers_time.clear()
            self._timers_log.clear()
        else:
            # Сбрасываем конкретный счетчик
            self._timers_time.pop(timer, 0.0)
            self._timers_log.pop(timer, "")

    def save(self, timer: TTimerType, time: float, log: str = "") -> None:
        """Сохраняет время и лог таймера и регистрирует таймер как последний сохраненный"""
        self._timers_time[timer] = time
        self._timers_log[timer] = log
        self._last_timer = timer

    def time(self, timer: TTimerType | None = None) -> float:
        """Возвращает сохраненное время заданного таймера. Если таймер не задан (None),
        возвращает время последнего сохраненного таймера"""
        if timer is None:
            timer = self._last_timer
        return self._timers_time.get(timer, 0.0)

    def log(self, timer: TTimerType | None = None) -> str:
        """Возвращает сохраненный текст лога заданного таймера. Если таймер не задан (None),
        возвращает лог последнего сохраненного таймера"""
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
        is_show (bool): Флаг, разрешающий вывод результатов замера. Default: True
        repeat (int): Количество повторных запусков вызываемого объекта. Default: 1000
        logger (Logger | str | None): Вывод результатов замеров. По-умолчанию на консоль. Default: None
    """

    # Дескрипторы
    time_source = TypeCallable()
    is_accumulate = TypeBool()
    is_show = TypeBool()
    repeat = TypePosInteger()
    logger = TypeLogger()

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
        self.logger = logger
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
        Измеренное время доступно через метод time. Неявно можно передать настроечные
        параметры time_source, repeat, is_accumulate, is_show и logger в виде именованных аргументов,
        иначе будут использованы значения соответствующих свойств экземпляра класса.
        Неявные настроечные параметры действуют единоразово только на момент вызова метода и
        не меняют параметры, заданные при инициализации экземпляра класса Timers.
        В режимах "Timer" и "Call" параметр repeat не используется.
        В режимах "Total", "Best" и "Average" параметр is_accumulate игнорируется.

        Examples:
            >>> instance(func, *args, **kwargs)
            >>> instance(func, *args, **kwargs, timer='Best', repeat=100)
            >>> instance(func, *args, **kwargs, time_source=time.process_time, is_show=False)

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
            result: Результат выполнения вызываемого объекта func.
            RuntimeError: Если во время вызова исполняемого объекта произошла ошибка, генерируется исключение с
            указанием имени вызываемого объекта и его аргументов.

        Returns:
            Any: Результат, возвращаемый вызываемым объектом.
        """
        # Формируем список параметров, которые передаются в метод __init__.
        # Для каждого из параметров выполняем три шага:
        # - пытаемся получить значение из входных параметров метода __call__ из kwargs
        # - иначе пытаемся получить значение из атрибута экземпляра класса Timers
        # - иначе подставляем значение по-умолчанию, заданное в методе __init__
        # Если значение по-умолчанию не задано, подставляется None
        # В итоге будет сформирован словарь из атрибутов класса Timers и их значений
        _attrs = {
            attr: kwargs.pop(attr, val) for attr, val in self._get_init_parameters()
        }
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
                        _time += self.time(timer)
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
        # Собираем из списка параметров метода __init__ строку инициализации экземпляра класса
        # с текущими значениями параметров, а не с переданными в момент инициализации.
        return "".join(
            (
                f"{type(self).__name__}(",
                ", ".join(
                    # Отбираем те параметры, которые удалось идентифицировать
                    f"{attr} = {demo.get_object_modname(value)}"
                    for attr, value in self._get_init_parameters(
                        empty_value=_not_defined
                    )
                    if value is not _not_defined
                ),
                ")",
            )
        )

    def __str__(self) -> str:
        """По-умолчанию возвращает лог последнего сохраненного таймера"""
        return self.__timers.log()

    def log(self, timer: TTimerType | None = None) -> str:
        """Возвращает лог заданного таймера или последнего сохраненного"""
        return self.__timers.log(timer)

    # Для поддержки протокола менеджера контекста, реализованы методы __enter__ и __exit__
    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        # Внутри контекстного менеджера возможен вызов self.reset()
        if self.__start_time is not None:
            self.stop()

    def __getattr__(self, attr_name):
        if attr_name in get_args(TTimerType):
            return self.time(attr_name)
        return super().__getattribute__(attr_name)

    def register(self, func: Callable):
        """Позволяет использовать класс как декоратор: @Timers().register. При этом можно задать
        функцию источника времени и другие параметры для класса Timers.
        Допускается использовать один и тот же таймер и как декоратор, и как менеджер контекста и как ручной таймер.
        При этом декоратор может быть вложен в менеджер контента или в ручной таймер.

        Например:

        >>> @Timers(time.process_time, is_accumulate=True, is_show=False, logger=MyLogger).register
        Альтернатива, предварительно создать экземпляр класса Timers, настроить необходимые параметры и
        использовать экземпляр как декоратор:
        >>> @instance.register.

        Параметр repeat для декоратора не используется и игнорируется.
        """

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return self(func, *args, **kwargs, timer="Call")

        return _wrapper

    def _get_init_parameters(
        self, empty_value: Any = None
    ) -> Iterator[tuple[str, Any]]:
        """Формирует итератор параметров и их значений метода __init__ экземпляра класса
        с текущими значениями параметров, а не с переданными в момент инициализации.
        Важно! Имя атрибута экземпляра класса должно совпадать с именем параметра
        кроме начальных подчеркиваний.

        Args:
            empty_value: Значение по-умолчанию для параметров не найденных в экземпляре класса. Defaults to None.

        Returns:
            Iterator[tuple[str, Any]]: Итератор кортежей (параметр, значение)
        """
        return (
            (
                attr,
                getattr(
                    self,
                    attr,  # Атрибут: self.somename
                    getattr(
                        self,
                        "_" + attr,  # Атрибут: self._somename
                        getattr(
                            self,
                            # Альтернатива:  f"_{self.__class__.__name__}__{arg}"
                            f"_{type(self).__name__}__{attr}",  # Атрибут: self.__somename
                            val.default  # Значение переданное в __init__
                            if val.default is not val.empty
                            else empty_value,  # Значение-пустышка в случае неудачного поиска
                        ),
                    ),
                ),
            )
            for attr, val in inspect.signature(self.__init__).parameters.items()
        )

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
        func_parameters = func_signature.parameters.keys()
        # Переупаковываем словарь именованных аргументов, оставляя только те, что заданы
        # в сигнатуре вызываемого объекта, т.к. дополнительно в списке именованных
        # аргументов могут присутствовать аргументы специфичные для класса Timers()
        kwargs = {k: v for k, v in kwargs.items() if k in func_parameters}

        # Формируем строку аргументов и их значений, переданных в вызываемый объект
        try:
            _str: str = ", ".join(
                f"{arg_name}={arg_value!r}"
                for arg_name, arg_value in func_signature.bind(
                    *args, **kwargs
                ).arguments.items()
            )
        except Exception:
            _str = ""
        return _str

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
                # cast нужен исключительно для успокоения статического анализатора типов
                _logger = cast(logging.Logger, self.logger)
                _logger.info(_log)

        return self.time(_timer)

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
    def is_running(self) -> bool:
        """Признак работающего таймера"""
        return self.__start_time is not None

    def time(self, timer: TTimerType | None = None) -> float:
        """Возвращает время заданного таймера или по-умолчанию последнего сохраненного
        Поддерживается вызов таймера в качестве атрибута экземпляра класса.

        Например:
        >>> instance.Timer
        >>> instance.Call
        >>> instance.Total
        >>> instance.Best
        >>> instance.Average

        что приведет к вызову метода instance.time() с заданным таймером (см. __getattr__)."""
        if timer == "Timer" and self.is_running:
            raise TimerNotStopError
        return self.__timers.time(timer)


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
    tmr(countdown, 50000, t=10, repeat=10, timer="Best")

    pass
