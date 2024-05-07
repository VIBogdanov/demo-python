from assistools import get_ranges_index
from puzzles import (
    get_combination_numbers,
    get_number_permutations,
    get_pagebook_number,
)
from sundry import (
    find_intervals,
    find_item_by_binary,
    find_item_by_interpolation,
    find_nearest_number,
    get_common_divisor,
    sort_by_bubble,
    sort_by_merge,
    sort_by_merge2,
    sort_by_selection,
    sort_by_shell,
)

if __name__ == "__main__":
    print(
        "\n- Формирует список индексов диапазонов, на которые можно разбить список заданной длины."
    )
    print(" get_ranges_index(50, 10) -> ", end="")
    for res in get_ranges_index(50, 10):
        print(tuple(res), end=" ")
    print("")

    print(
        "\n- Функция нахождения наибольшего общего делителя двух целых чисел без перебора методом Евклида."
    )
    print(f" get_common_divisor(20, 12) -> {get_common_divisor(20, 12)}")

    print("\n- Поиск элемента в массиве данных при помощи бинарного алгоритма.")
    print(
        f" find_item_by_binary([-20, 30, 40, 50], 30) -> {find_item_by_binary([-20, 30, 40, 50], 30)}"
    )

    print("\n- Поиск элемента в массиве данных при помощи алгоритма интерполяции.")
    print(
        f" find_item_by_interpolation([-1, -2, 3, 4, 5], 4) -> {find_item_by_interpolation([-1, -2, 3, 4, 5], 4)}"
    )

    print(
        "\n- Поиск в списке из чисел последовательного непрерывного интервала(-ов) чисел, сумма которых равна искомому значению."
    )
    print(" find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0) -> ", end="")
    for res in find_intervals([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], 0):
        print(tuple(res), end=" ")
    print("")

    print(
        "\n- Поиск ближайшего целого числа, которое меньше или больше заданного и состоит из тех же цифр."
    )
    print(f" find_nearest_number(273145) -> {find_nearest_number(273145)}")

    print("\n- Сортировки методом пузырька.")
    print(
        f" sort_by_bubble([2, 7, 3, 1, 4, 5]) -> {sort_by_bubble([2, 7, 3, 1, 4, 5])}"
    )

    print("\n- Сортировки методом слияния.")
    print(f" sort_by_merge([2, 7, 3, 1, 4, 5]) -> {sort_by_merge([2, 7, 3, 1, 4, 5])}")

    print("\n- Усовершенствованная версия сортировки методом слияния.")
    print(
        f" sort_by_merge2([2, 7, 3, 1, 4, 5]) -> {sort_by_merge2([2, 7, 3, 1, 4, 5])}"
    )

    print("\n- Сортировки методом Shell.")
    print(
        f" sort_by_shell([2, 7, 3, 1, 4, 5]) -> {sort_by_shell([2, 7, 3, 1, 4, 5], method = 'Shell')}"
    )

    print("\n- Сортировки методом выбора.")
    print(
        f" sort_by_selection([2, 7, 3, 1, 4, 5]) -> {sort_by_selection([2, 7, 3, 1, 4, 5])}"
    )

    print(
        "\n- Минимальное количество перестановок, которое необходимо произвести для выравнивания списков."
    )
    print(
        f" get_number_permutations([10, 31, 15, 22, 14, 17, 16], [16, 22, 14, 10, 31, 15, 17])) -> {get_number_permutations([10, 31, 15, 22, 14, 17, 16], [16, 22, 14, 10, 31, 15, 17])}"
    )

    print("\n- Сформировать все возможные уникальные наборы чисел из указанных цифр.")
    print(f" get_combination_numbers([2, 7]) -> {get_combination_numbers([0, 2, 7])}")

    print("\n- Олимпиадная задача. См. описание в puzzles.py.")
    print(f" get_pagebook_number(27, 2, [8,0]) -> {get_pagebook_number(27, 2, [8,0])}")

    print("")
    pass
