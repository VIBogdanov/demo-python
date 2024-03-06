from assistools import get_ranges_index
from sundry import (
    find_intervals,
    find_item_by_binary,
    find_item_by_interpolation,
    get_common_divisor,
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

    print("")
    pass
