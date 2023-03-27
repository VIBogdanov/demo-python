# demo-python
Python code examples

Представленные в данном репозитории утилиты созданы для целей демонстрации
стиля разработки на Python.
В следсвии чего код перегружен комментариями для описания каждого шага разработки.
При этом код является полностью рабочим и может быть использована в проектах.

Перечень модулей и краткое описание:

Permutator
Поиск ближайшего большего или меньшего значения для целого числа, при этом
искомое значение должно состоять из того же набора цифр.
Использован алгоритм попарной перестановки чисел.
В модуле реализован механизм мультизадачности, но его практическая ценность
ничтожна, т.к. затраты на создание и запуск пула мультипроцессов ощутимо превышают
затраты на последовательное выполнение алгоритма. В итоге, механизм мультизадачности
реализован сугубо в демонстрационных целях и по-умолчанию отключен.