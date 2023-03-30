from collections import defaultdict
from itertools import groupby
from typing import Union


def count_items(
    data_list: list,
    key: str,
    operation: str = "Total",
) -> Union[int, float, None]:
    """
    Функция подсчета количества одинаковых элементов в списке элементов (по-умолчанию).
    При этом список содержит неотсортированные и разнотипные элементы.
    Кроме общего количества элементов, возможно получить минимальный или максимальный
    размер группы или подсчитать количество групп.

    Args:
        data_list (list): Список неупорядоченных разнотипных элементов.

        key (str): Имя элемента в виде строки, для которого выполняется подсчет.

        operation (str, optional): Вид результирующего подсчета. Defaults to "Total".

    Returns:
        Union[int, float, None]: Результат подсчета в зависимости от заданного вида.

    Example:
    >>> data = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 'a', 'a', 'a']
        count_items(data, '0')
        9
        count_items(data, '0', 'min')
        1
        count_items(data, '0', 'max')
        5
        count_items(data, '0', 'count')
        4

    """
    # словарь группирующий элементы из data_list
    dict_data: defaultdict = defaultdict(list)
    # словарь с перечнем операций над элементами групп
    dict_operations: dict = {
        # общее количество элементов во всех группах
        "total": lambda: sum(dict_data.get(key, [0])),
        # минимальный размер группы
        "min": lambda: min(dict_data.get(key, [0])),
        # максимальный размер группы
        "max": lambda: max(dict_data.get(key, [0])),
        # количество групп
        "count": lambda: len(dict_data.get(key, [0])),
        # ... возможно дальнейшее расширение списка операций
    }
    # список ключей из элементов и количество элементов в каждой группе
    groups_list: list = [(str(k), len(tuple(v))) for k, v in groupby(data_list)]
    for k, v in groups_list:
        dict_data[k].append(v)
    # подсчитываем значение в зависимости от запрошенной операции
    # если вид операции задан некорректно, возвращается None
    return dict_operations.get(operation.lower(), lambda: None)()


if __name__ == "__main__":
    data = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 'a', 'a', 'a']
    print(count_items(data, '0'))
