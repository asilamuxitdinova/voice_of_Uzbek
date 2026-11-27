"""Uzbek Latin <-> Cyrillic transliteration (MMS TTS reads Cyrillic; scoring needs Latin).

    python translit.py "O'zbekiston go'zal mamlakat"   ->  Ўзбекистон гўзал мамлакат

Rule-based; handles the common cases (oʻ, gʻ, sh, ch, yo/yu/ya/ye, word-initial e -> э,
the tutuq belgisi ' -> ъ). Loanwords with ts/ц etc. are left as-is.
"""
import re
import sys

APOSTROPHES = ("ʻ", "’", "‘", "`", "ʼ", "´")

DIGRAPHS = {
    "o'": "ў", "g'": "ғ",
    "sh": "ш", "ch": "ч",
    "yo": "ё", "yu": "ю", "ya": "я", "ye": "е",
}

SINGLE = {
    "a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ", "i": "и",
    "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "x": "х",
    "y": "й", "z": "з", "c": "к", "w": "в",
}


def _case(src: str, cyr: str) -> str:
    return cyr.upper() if src[0].isupper() else cyr


def to_cyrillic(text: str) -> str:
    for a in APOSTROPHES:
        text = text.replace(a, "'")
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        two = text[i:i + 2].lower()
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < n else ""

        if two in DIGRAPHS:
            out.append(_case(ch, DIGRAPHS[two]))
            i += 2
            continue

        low = ch.lower()
        if low == "e":
            word_start = not (prev.isalpha() or prev == "'")
            out.append(_case(ch, "э" if word_start else "е"))
        elif low in SINGLE:
            out.append(_case(ch, SINGLE[low]))
        elif ch == "'" and prev.isalpha() and nxt.isalpha():
            out.append("ъ")  # tutuq belgisi, e.g. ma'no -> маъно
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def needs_conversion(text: str) -> bool:
    """True if the text is mostly Latin letters."""
    latin = len(re.findall(r"[A-Za-z]", text))
    cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    return latin > cyr


CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "'", "ь": "",
    "ы": "i", "э": "e", "ё": "yo", "ю": "yu", "я": "ya",
    "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h",
    # Kazakh-style letters that stock Whisper sometimes produces for Uzbek audio
    "ұ": "u", "ү": "u", "ң": "ng", "ә": "a", "і": "i", "ө": "o'", "һ": "h",
}
_CYR_VOWELS = set("аеёиоуўэюяыұүәіө")


def to_latin(text: str) -> str:
    """Uzbek Cyrillic -> Latin. е is "ye" at the start of a word or after a vowel/ъ/ь."""
    out = []
    for i, ch in enumerate(text):
        low = ch.lower()
        if low == "е":
            prev = text[i - 1].lower() if i > 0 else ""
            lat = "ye" if (not prev.isalpha() or prev in _CYR_VOWELS or prev in "ъь") else "e"
        elif low in CYR2LAT:
            lat = CYR2LAT[low]
        else:
            out.append(ch)
            continue
        if ch.isupper() and lat:
            lat = lat[0].upper() + lat[1:]
        out.append(lat)
    return "".join(out)


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "O'zbekiston go'zal mamlakat. Ertaga shanba, ma'no, yangi choy."
    cyr = to_cyrillic(sample)
    print(cyr)
    print(to_latin(cyr))
