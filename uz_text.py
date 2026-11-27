"""Uzbek text normalization for fair ASR scoring.

Two normalizers:
  norm_basic(text) - lowercase, unify apostrophes, drop punctuation (the original scoring)
  norm_fair(text)  - norm_basic + Cyrillic->Latin + numbers written as words, so that
                     "15%" == "o'n besh foiz", "2026-yil" == "ikki ming yigirma oltinchi yil",
                     "бор" == "bor". Only differences in *spelling convention* are removed;
                     real recognition errors still count.

    python uz_text.py "Buyurtma 15-yanvarda 240 000 so‘mga, 15% chegirma bilan"
"""
import re
import sys

from translit import to_latin

APOSTROPHES = ("ʻ", "’", "‘", "`", "ʼ", "´", "ʹ")

ONES = ["nol", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
TENS = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]
SCALES = [(10**9, "milliard"), (10**6, "million"), (1000, "ming")]
VOWELS = set("aeiou")
ABBREVIATIONS = {
    "km": "kilometr", "sm": "santimetr", "mm": "millimetr", "kg": "kilogramm",
    "mln": "million", "mlrd": "milliard",
}


def _below_1000(n: int) -> list:
    words = []
    h, rest = divmod(n, 100)
    if h:
        words += ([ONES[h]] if h > 1 else []) + ["yuz"]
    t, o = divmod(rest, 10)
    if t:
        words.append(TENS[t])
    if o:
        words.append(ONES[o])
    return words


def number_to_words(n: int) -> str:
    """240000 -> "ikki yuz qirq ming". Numbers >= 10^12 are returned as digits."""
    if n == 0:
        return "nol"
    if n >= 10**12:
        return str(n)
    words = []
    for value, name in SCALES:
        q, n = divmod(n, value)
        if q:
            words += ([] if (value == 1000 and q == 1) else _below_1000(q)) + [name]
    return " ".join(words + _below_1000(n))


def ordinal(words: str) -> str:
    """"besh" -> "beshinchi", "ikki" -> "ikkinchi", "yigirma" -> "yigirmanchi"."""
    head, _, last = words.rpartition(" ")
    last = last + ("nchi" if last[-1] in VOWELS else "inchi")
    return f"{head} {last}".strip()


_NUM = r"\d{1,3}(?:[  ]\d{3})+|\d+"  # "240 000" (space as thousands separator) or "2026"


def _int(s: str) -> int:
    return int(re.sub(r"\D", "", s))


def numbers_to_words(text: str) -> str:
    # percent: 15% -> o'n besh foiz
    text = re.sub(rf"({_NUM})\s*%", lambda m: f" {number_to_words(_int(m.group(1)))} foiz ", text)
    # explicit ordinal suffix: 15-chi / 15-inchi / 2-nchi -> o'n beshinchi
    text = re.sub(rf"({_NUM})\s*-\s*(?:inchi|nchi|chi)\b",
                  lambda m: f" {ordinal(number_to_words(_int(m.group(1))))} ", text)
    # number + hyphen + word is an ordinal in Uzbek: 5-sinf, 2026-yil, 15-yanvar
    text = re.sub(rf"({_NUM})\s*-(?=[^\W\d_])",
                  lambda m: f" {ordinal(number_to_words(_int(m.group(1))))} ", text)
    # plain cardinals
    return re.sub(_NUM, lambda m: f" {number_to_words(_int(m.group(0)))} ", text)


def norm_basic(text: str) -> str:
    """Lowercase, unify the oʻ/gʻ apostrophes, drop punctuation."""
    text = text.lower()
    for a in APOSTROPHES:
        text = text.replace(a, "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_fair(text: str) -> str:
    text = text.lower()
    for a in APOSTROPHES:
        text = text.replace(a, "'")
    if re.search(r"[Ѐ-ӿ]", text):
        text = to_latin(text)
    text = numbers_to_words(text)
    text = norm_basic(text)
    # unit abbreviations are spoken in full
    text = re.sub(r"\b(" + "|".join(ABBREVIATIONS) + r")\b", lambda m: ABBREVIATIONS[m.group(1)], text)
    # "bir yuz" / "bir ming" and "yuz" / "ming" are both used in speech
    text = re.sub(r"\bbir (yuz|ming)\b", r"\1", text)
    return text


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "Buyurtma 15-yanvarda 240 000 so‘mga, 15% chegirma bilan. Бугун 2026-йил."
    print("basic:", norm_basic(sample))
    print("fair: ", norm_fair(sample))
