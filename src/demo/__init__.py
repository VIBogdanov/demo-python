# Должно быть: from .assistools import get_positive_int, is_int, type_checking ...
# from . import assistools - альтернативный вариант, но требует обращения через demo.assistools
# Умышленно в demo импортируем все имена, дабы иметь возможность обращаться через просто demo
from .assistools import *
from .puzzles import *
from .sundry import *
from .timers import MiniTimers, Timers
