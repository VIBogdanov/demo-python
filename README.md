# demo-python
## **Python code examples**
Представленные в данном репозитории утилиты созданы для целей демонстрации
стиля разработки на Python.
В следсвие чего код перегружен комментариями для описания каждого шага разработки.
При этом код является полностью рабочим и может быть использована в проектах.
Весь код написан и протестирован на Python 3.11.2

#### <u>Перечень модулей и краткое описание:</u>

### **permutator**  
Поиск ближайшего большего или меньшего значения для целого числа, при этом
искомое значение должно состоять из того же набора цифр.
Использован алгоритм попарной перестановки чисел.
В модуле реализован механизм мультизадачности, но его практическая ценность
ничтожна, т.к. затраты на создание и запуск пула мультипроцессов ощутимо превышают
затраты на последовательное выполнение алгоритма. В итоге, механизм мультизадачности
реализован сугубо в демонстрационных целях и по-умолчанию отключен.

### **mult_matrix**  
Поэлементное перемножение в каждой строке двумерной матрицы с учетом органичений (min-max).
Получаем массив значений как результат поэлементного перемножения. Размер результирующего
массива соответствует количеству строк матрицы.

### **count_items**  
Подсчет количества одинаковых элементов в неотсортированном и неупорядоченном списке.
По-умолчанию подсчитывается общее количество искомого элемента, заданного в виде ключа. Помимо
общего количества, возможно получить минимальный или максимальный размер групп, состоящих из искомого элемента, или подсчитать количество этих групп.

### **sundry**  
Набор различных функций реализующие алгоритмы сортировки, поиска, фильтрации и т.п. Каждая функция имеет подробное описание и готова к применению.
- ***find_intervals*** - Поиск в списке из чисел последовательного непрерывного интервала(-ов) чисел,сумма которых равна искомому значению.
- ***find_nearest_number*** - Поиск ближайшего целого числа, которое меньше или больше заданного и состоит из того же набора цифр. Использован алгоритм попарной перестановки чисел. В функции реализован механизм мультизадачности, но его практическая ценность ничтожна, посему по-умолчанию отключен.
- ***find_item_by_binary*** - Поиск элемента в массиве данных при помощи бинарного алгоритма. При поиске учитывается направление сортировки массива.
- ***sort_by_bubble*** - Сортировка списка данных методом "пузырька". В отличии от классического, метод усовершенствован одновременным поиском как максимального, так и минимального значений. Реализована двунаправленная сортировка.
- ***sort_by_merge*** - Сортировка списка данных методом слияния. Реализована двунаправленная сортировка.
- ***sort_by_shell*** - Сортировка списка методом Shell. Кроме классического метода формирования диапазона чисел для перестановки, возможно использовать следующие методы: Hibbard, Sedgewick, Knuth, Fibonacci. Реализована двунаправленная сортировка.
- ***sort_by_selection*** - Сортировка списка методом выбора. Это улучшенный вариант пузырьковой сортировки,за счет сокращения числа перестановок элементов. В отличии от классического метода, добавлена возможность одновременного поиска максимального и минимального элементов на каждой итерации. Реализована двунаправленная сортировка.
