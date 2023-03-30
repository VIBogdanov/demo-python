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