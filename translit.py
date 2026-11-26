"""Uzbek Latin -> Cyrillic transliteration (needed because MMS TTS reads Cyrillic).

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


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "O'zbekiston go'zal mamlakat. Ertaga shanba, ma'no, yangi choy."
    print(to_cyrillic(sample))
