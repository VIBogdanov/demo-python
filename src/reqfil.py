"""
Модуль обработки csv-файла с миллионом записей, состоящих из текста запросов
пользователей и количество этих запросов за определенный период времени.
На входе задается поисковая фраза и метод поиска поисковой фразы в запросе.
В поисковой фразе можно задавать якорные и сопутствующие слова.
Любое слово, начинающееся с заглавной буквы, считается якорным.
Особенность якорных слов - они комбинируются с сопутствующими и всегда
присутствуют в комбинациях поиска.

Например: в поисковой фразе "Платье женское летнее" якорное слово - "Платье".
Поиск будет производится по следующим комбинациям:
["платье", "платье женское", "платье летнее", "платье женское летнее"],
т.е. якорное слово "Платье" всегда присутствует в поиске, а сопутствующие слова могут
отсутствовать вовсе или присутствовать в различных комбинациях.
Если задать поисковую фразу "платье женское летнее" или "Платье Женское Летнее",
то поиск будет производится по единственной комбинации ["платье женское летнее"]

Внимание!!! Используются две сторонние библиотеки: pandas и openpyxl
"""

from collections.abc import Callable, Generator
from enum import Enum, auto
from itertools import chain, combinations
from pathlib import Path
from typing import NamedTuple

from pandas import DataFrame, ExcelWriter

import demo

STRIP_TEMPL = r"""_ .,:;"'!?-+=*()[]{}\|/"""
BUFFER_SIZE = 1024 * 100


# Перечисление методов сравнения поисковой фразы с запросом
class CompareType(str, Enum):
    Full = auto()  # Полное совпадение
    Sub = auto()  # Поисковая фраза содержится в запросе
    Any = auto()  # Хотя бы одно любое слово из поисковой фразы содержится в запросе
    All = auto()  # Поисковая фраза игнорируется


# Структура для извлеченного из файла запроса
class Request(NamedTuple):
    request: str  # Оригинальный текст запроса
    quantity: int  # Количество запросов за период
    words: set[str]  # Список очищенных слов, из которых состоит запрос


# Поисковая фраза после обработки
class Phrase(NamedTuple):
    # Комбинации слов, из которых состоит поисковая фраза
    combo_words: list[set[str]]
    compare_type: CompareType  # Выбранный метод сравнения


def get_request(csv_filename: str | Path) -> Generator[Request, None, None]:
    """Функция-генератор считывает записи из csv-фала и производит
    следующую обработку:
    - Разделяет записи на текст запроса и на количество запросов за период
    - Разбирает текст запроса на отдельные слова
    - Очищает список слов от небуквенных знаков

    При этом файл не загружается в память целиком, а считывается построчно.

    Args:
        csv_filename (str | Path): Полный путь к csv-фалу, включая имя самого файла

    Yields:
        Generator[Request]: Возвращает структуру Request,
        содержащей запрос, количество и список слов
    """
    with open(
        csv_filename, encoding="utf-8-sig", mode="rt", buffering=BUFFER_SIZE
    ) as csv_file:
        for line in csv_file:
            # пытаемся выделить из строки текст запроса и его количество
            try:
                req_str, qty_str = line.rsplit(",", maxsplit=1)
                qty_int = int(qty_str)
            except (ValueError, TypeError):
                # В случае сбоя считаем что запрос встречается хотя бы один раз
                req_str, qty_int = line, 1
            # возвращаем оригинальный запрос, его кол-во и список слов запроса, очищенных от мусора
            yield Request(
                req_str,
                qty_int,
                set(word.strip(STRIP_TEMPL).lower() for word in req_str.split()),
            )


def get_phrase(
    phrase_string: str, compare_type: CompareType = CompareType.Any
) -> Phrase:
    """Разбирает поисковую фразу на отдельные слова и создает комбинации из
    якорных слов и сопутствующих.

    Args:
        phrase_string (str): Поисковая фраза
        compare_type (CompareType, optional): Тип метода сравнения. Defaults to CompareType.Any.

    Returns:
        Phrase: Возвращает структуру из комбинации слов поисковой фразы и метода сравнения.
    """
    # Результирующая комбинация якорных и сопутствующих слов
    phrase_combo_words: list[set[str]] = list()

    # раскладываем поисковую фразу на отдельные слова и очищаем их
    # используем генератор для будущего однократного прохода без сохранения промежуточных данных
    phrase_words_clear: Generator[str, None, None] = (
        word.strip(STRIP_TEMPL) for word in phrase_string.split()
    )

    match compare_type:
        case CompareType.Full | CompareType.Sub:
            # два отдельных set для якорных и сопутствующих слов
            # используются для комбинирования якорных слов с сопутствующими
            phrase_fix_words: set[str] = set()
            phrase_words: set[str] = set()

            for word in phrase_words_clear:
                if word[0].isupper():  # формируем список якорных слов
                    phrase_fix_words.add(word.lower())
                else:  # формируем список сопутствующих слов
                    phrase_words.add(word.lower())
            # Возможна ситуация когда одинаковое слово попадет в оба списка. Человеческий фактор
            phrase_words -= phrase_fix_words

            # если есть что комбинировать
            if phrase_words and phrase_fix_words:
                phrase_combo_words = list(
                    set(phrase_combo).union(phrase_fix_words)
                    for combo in (
                        combinations(phrase_words, i)
                        for i in range(1, (len(phrase_words) + 1))
                    )
                    for phrase_combo in combo
                )
                # Якорные слова обязательно должны присутствовать в списке комбинаций
                phrase_combo_words.append(phrase_fix_words)
            else:  # иначе просто подгружаем тот список слов, который не пуст
                phrase_combo_words.append(
                    phrase_words if phrase_words else phrase_fix_words
                )
        case CompareType.Any:
            # Просто добавляем список всех слов поисковой фразы. Комбинировать не нужно
            phrase_combo_words.append(set(word.lower() for word in phrase_words_clear))

    # В последнем случае (CompareType.All) посисковая фраза не используется. Возвращаем пустой список
    return Phrase(phrase_combo_words, compare_type)


def get_check_request(
    phrase: Phrase, min_quantity: int = 1
) -> Callable[[Request], bool]:
    """Формирует функцию, которая будет использована при отборе запросов из csv-фала.
    В зависимости от выбранного метода, применяются следующие механизмы отбора:
    - Full - полное совпадение поисковой фразы с запросом. Например: "платье женское" == "платье женское"
    При этом порядок слов в запросе не важен: "платье женское" == "женское платье"
    - Sub - запрос должен содержать поисковую фразу: "платье женское" == "платье женское летнее"
    - Any - запрос должен содержать хотя бы одно (или более) любое слово из поисковой фразы.
    Например: "платье женское" == "платье с длинными рукавами" или "платье женское" == "пальто женское зимнее"
    - All - полностью игнорируется поисковая фраза. Учитывается только параметр min_quantity, который
    отфильтровывает низкочастотные запросы. Например: при min_quantity = 1000 в выборку попадут запросы
    с частотой появления за период не ниже 1000 включительно.

    Args:
        phrase (Phrase): Структура с обработанной поисковой фразой, содержащая комбинации слов поисковой фразы и метод сравнения.

        min_quantity (int, optional): Минимальная частота появления запроса за период. Defaults to 1 - все запросы.

    Returns:
        Callable[[Request], bool]: Функция, сравнивающая поисковую фразу с запросом и возвращающая True,
        если сравнение успешно.
    """
    match phrase.compare_type:
        # Поисковая фраза полностью совпадает с текстом запроса. Порядок слов не важен.
        case CompareType.Full:

            def fn(request: Request) -> bool:
                return (
                    request.words in phrase.combo_words
                    and request.quantity >= min_quantity
                )
        # Поисковая фраза содержится в тексте запроса
        case CompareType.Sub:

            def fn(request: Request) -> bool:
                for phrase_combo_words in phrase.combo_words:
                    if (
                        phrase_combo_words.issubset(request.words)
                        and request.quantity >= min_quantity
                    ):
                        return True
                return False
        # Если в запросе встречается хотя бы одно любое слово из поисковой фразы
        case CompareType.Any:

            def fn(request: Request) -> bool:
                return (
                    not phrase.combo_words[0].isdisjoint(request.words)
                    and request.quantity >= min_quantity
                )
        # Игнорируем поисковую фразу и выводим весь список запросов с учетом минимального количества
        case CompareType.All:

            def fn(request: Request) -> bool:
                return request.quantity >= min_quantity
        # Что то пошло не так
        case _:
            # Формируем функцию-пустышку с той же сигнатурой, что у полноценной функции
            def nofn(_: Request) -> bool:
                return False

            # признак невозможности определить функцию сравнения.
            nofn.is_OK = False
            return nofn

    fn.is_OK = True
    return fn


def main():
    # Для данной демонстрации отсутствует возможность задавать параметры через командную строку.
    # Локальные константы
    SEARCH_PHRASE = "Платье платье женское леТнее"
    MIN_QUANTITY: int = 1  # Позволяет отфильтровать малочисленные запросы.
    COMPARE_TYPE = CompareType.Full

    DIR_NAME = r"../data"
    CSV_NAME = r"requests.csv"

    IS_SAVETOEXCEL = False
    IS_SHOWDATAFRAME = True

    # Полный путь до csv-файла. В данном случае считается, что файл csv храниться
    # в подпапке 'data' родительского каталога для текущего фала на один уровень выше.
    # Например: если текущий файл хранится в папке /samefolders/src/reqfil.py,
    # то csv файл должен быть расположен в папке /samefolders/data/requests.csv
    # Чтобы работать в текущей папке: DIR_NAME = r""
    csv_filename = Path(Path(__file__).parent, DIR_NAME, CSV_NAME).resolve()
    if not csv_filename.exists():
        demo.WarningToConsole(f"The file {csv_filename} does not exists!")
        return
    # Файл Excel сохраняем рядом с csv в той же папке с тем же именем
    excel_filename = csv_filename.with_suffix(".xlsx")

    # определяем функцию сравнения запросов с поисковой фразой
    check_request = get_check_request(
        get_phrase(SEARCH_PHRASE, COMPARE_TYPE), MIN_QUANTITY
    )
    # Если удалось определить функцию сравнения. Иначе программа завершиться досрочно без какой-либо обработки.
    if check_request.is_OK:
        # формируем генератор для построчного отбора запросов по поисковой фразе
        requests_generator: Generator[Request, None, None] = (
            request for request in get_request(csv_filename) if check_request(request)
        )
        # Т.к. генератор возвращает именованную структуру NamedTuple похожую на DataClass,
        # то возможно создание DataFrame без промежуточного словаря прямо из генератора
        dfr = DataFrame(requests_generator)
        if IS_SHOWDATAFRAME:
            print(dfr[["request", "quantity", "words"]] if len(dfr) else dfr.info())

        # Если выборка запросов не пуста
        if len(dfr) > 0:
            # Подсчитываем встречаемость слов в отобранных запросах
            dfw: DataFrame = (
                DataFrame(
                    {"words": chain.from_iterable(dfr["words"].values), "count": 1}
                )
                .groupby(["words"], as_index=False, sort=False)
                .sum()
            )
            # Сортируем по убыванию. Сначала самые популярные слова
            dfw.sort_values(by=["count"], ascending=False, inplace=True)

            # Сохраняем оба полученных DataFrame в Excel на отдельных листах с перезаписью
            if IS_SAVETOEXCEL:
                with ExcelWriter(excel_filename) as xls_writer:
                    dfr[["request", "quantity"]].to_excel(
                        xls_writer, sheet_name="requests", index=False
                    )
                    dfw.to_excel(xls_writer, sheet_name="words", index=False)
        else:
            demo.WarningToConsole("The request selection is empty!")
    else:
        demo.WarningToConsole("Function 'check_request' not defined!")


if __name__ == "__main__":
    main()
