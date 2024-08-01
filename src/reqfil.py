from collections.abc import Callable, Generator
from enum import Enum, auto
from itertools import chain, combinations
from pathlib import Path
from typing import NamedTuple

from pandas import DataFrame, ExcelWriter

STRIP_TEMPL = ' .,:;"!?-=()[]'


class CompareType(str, Enum):
    Full = auto()
    Sub = auto()
    Any = auto()
    All = auto()


class Request(NamedTuple):
    request: str
    quantity: int
    words: set[str]


class Phrase(NamedTuple):
    combo_words: list[set[str]]
    compare_type: CompareType


def get_request(csv_filename: str | Path) -> Generator[Request, None, None]:
    with open(
        csv_filename, encoding="utf-8-sig", mode="rt", buffering=102400
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
                set(map(lambda x: x.strip(STRIP_TEMPL).lower(), req_str.split())),
            )


def get_phrase(
    phrase_string: str, compare_type: CompareType = CompareType.Any
) -> Phrase:
    # Результирующая комбинация якорных и сопутствующих слов
    phrase_combo_words: list[set[str]] = list()

    # раскладываем поисковую фразу на отдельные слова и очищаем их
    phrase_string_clear: Generator[str, None, None] = (
        word for word in map(lambda w: w.strip(STRIP_TEMPL), phrase_string.split())
    )

    match compare_type:
        case CompareType.Full | CompareType.Sub:
            phrase_fix_words: set[str] = set()
            phrase_words: set[str] = set()
            # Из общего списка слов выделяем якорные слова и сопутствующие.
            for word in phrase_string_clear:
                if word[0].isupper():  # формируем список якорных слов
                    phrase_fix_words.add(word.lower())
                else:  # формируем список сопутствующих слов
                    phrase_words.add(word.lower())

            # print(phrase_fix_words)
            # Возможна ситуация когда одно и тоже слово попадет в оба списка. Человеческий фактор
            phrase_words -= phrase_fix_words
            # print(phrase_words)

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
                    phrase_words
                ) if phrase_words else phrase_combo_words.append(phrase_fix_words)
        case CompareType.Any:
            # Просто добавляем список слов поисковой фразы. Комбинировать не нужно
            phrase_combo_words.append(set(word.lower() for word in phrase_string_clear))
    # print(phrase_combo_words)
    # В последнем случае (CompareType.All) посисковая фраза не используется. Возвращаем пустой список
    return Phrase(phrase_combo_words, compare_type)


def get_check_request(
    phrase: Phrase, min_quantity: int = 1
) -> Callable[[Request], bool]:
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
        # Игнорируем поисковую фразу и выводим весь список запросов
        case CompareType.All:

            def fn(request: Request) -> bool:
                return request.quantity >= min_quantity
        # Что то пошло не так
        case _:

            def nofn(_: Request) -> bool:
                return False

            nofn.compare = None
            return nofn

    fn.compare = phrase.compare_type
    return fn


def main():
    # Локальные константы
    SEARCH_PHRASE = "Платье платье женское летнее"
    MIN_QUANTITY: int = 1
    COMPARE_TYPE = CompareType.Full

    DIR_NAME = r"data"
    CSV_NAME = r"requests.csv"

    csv_filename = Path(Path(__file__).parents[1], DIR_NAME, CSV_NAME)
    exel_filename = csv_filename.with_suffix(".xlsx")

    # определяем метод сравнения запросов с поисковой фразой
    check_request = get_check_request(
        get_phrase(SEARCH_PHRASE, COMPARE_TYPE), MIN_QUANTITY
    )
    # Если удалось определить метод сравнения
    if check_request.compare is not None:
        # формируем генератор для отбора запросов по поисковой фразе
        requests_generator: Generator[Request, None, None] = (
            request for request in get_request(csv_filename) if check_request(request)
        )
        # Т.к. генератор возвращает именованную структуру NamedTuple похожую на DataClass,
        # то возможно создание DataFrame без промежуточного словаря прямо из генератора
        dfr = DataFrame(requests_generator)
        if len(dfr) > 0:
            print(dfr[["request", "quantity", "words"]])
        else:
            print(dfr.info())
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
            # Для использования необходимо раскомментировать
            """
            with ExcelWriter(exel_filename) as xls_writer:
                dfr[["request", "quantity"]].to_excel(
                    xls_writer, sheet_name="requests", index=False
                )
                dfw.to_excel(xls_writer, sheet_name="words", index=False)
            """


if __name__ == "__main__":
    main()
