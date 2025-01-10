import editdistance as ed
import unicodedata
import json
import os
import re
import torch
from torch import Tensor
import numpy as np
from fractions import Fraction
from collections import defaultdict, Counter
from typing import Any, Callable, Mapping, Union, Iterator, List, Match, Optional
from more_itertools import windowed


ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )
    


class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"o", "oh", "zero"}
        self.ones = {
            name: i
            for i, name in enumerate(
                [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ],
                start=1,
            )
        }
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {
            name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()
        }
        self.tens_ordinal = {
            name.replace("y", "ieth"): (value, "th")
            for name, value in self.tens.items()
        }
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {
            name + "s": (value, "s") for name, value in self.multipliers.items()
        }
        self.multipliers_ordinal = {
            name + "th": (value, "th") for name, value in self.multipliers.items()
        }
        self.multipliers_suffixed = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(
            list(self.preceding_prefixers.values())
            + list(self.following_prefixers.values())
        )
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = set(
            [
                key
                for mapping in [
                    self.zeros,
                    self.ones,
                    self.ones_suffixed,
                    self.tens,
                    self.tens_suffixed,
                    self.multipliers,
                    self.multipliers_suffixed,
                    self.preceding_prefixers,
                    self.following_prefixers,
                    self.suffixers,
                    self.specials,
                ]
                for key in mapping
            ]
        )
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for prev, current, next in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                assert f is not None
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if (
                        prev in self.tens and ones < 10
                    ):  # replace the last zero with the digit
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # replace "<number> and a half" with "<number> point five"
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # write "one(s)" instead of "1(s)", just for the readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self):
        mapping_path = os.path.join("data/english.json")
        self.mapping = json.load(open(mapping_path))

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    def __init__(self):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep numeric symbols

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        return s


class EvaluationTokenizer(object):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    Args:
        tokenizer_type (str): the type of sacreBLEU tokenizer to apply.
        lowercase (bool): lowercase the text.
        punctuation_removal (bool): remove punctuation (based on unicode
        category) from text.
        character_tokenization (bool): tokenize the text to characters.
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)
    # ALL_TOKENIZER_TYPES = ChoiceEnum(["none", "13a", "intl", "zh", "ja-mecab"])

    def __init__(
        self,
        tokenizer_type: str = "13a",
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        from sacrebleu.tokenizers import TOKENIZERS

        assert tokenizer_type in TOKENIZERS, f"{tokenizer_type}, {TOKENIZERS}"
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(
            t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == "P" for c in t)
        )

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized


def compute_wer(hyps, refs) -> float:
    normalizer = EnglishTextNormalizer()

    norm_refs = [normalizer(ref) for ref in refs]
    norm_hyps = [normalizer(hyp) for hyp in hyps]
    
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    for i in range(len(norm_refs)):
        ref = norm_refs[i]
        pred = norm_hyps[i]

        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()

        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    
    wer = distance / ref_length

    print(f"WER: {wer*100:0.4f}%")




def cider_d(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    n: int = 4,
    sigma: float = 6.0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_tfidf: bool = False,
    scale: float = 10.0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    """Consensus-based Image Description Evaluation function.

    - Paper: https://arxiv.org/pdf/1411.5726.pdf

    .. warning::
        This metric requires at least 2 candidates with 2 sets of references, otherwise it will raises a ValueError.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param n: Maximal number of n-grams taken into account. defaults to 4.
    :param sigma: Standard deviation parameter used for gaussian penalty. defaults to 6.0.
    :param tokenizer: The fast tokenizer used to split sentences into words. defaults to str.split.
    :param return_tfidf: If True, returns the list of dictionaries containing the tf-idf scores of n-grams in the sents_score output.
        defaults to False.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    cooked_cands, cooked_mrefs = _cider_d_update(
        candidates,
        mult_references,
        n,
        tokenizer,
        [],
        [],
    )
    return _cider_d_compute(
        cooked_cands,
        cooked_mrefs,
        return_all_scores,
        n,
        sigma,
        return_tfidf,
        scale,
    )


def _cider_d_update(
    candidates: list[str],
    mult_references: list[list[str]],
    n: int,
    tokenizer: Callable[[str], list[str]],
    prev_cooked_cands: list[Counter],
    prev_cooked_mrefs: list[list[Counter]],
) -> tuple[list, list]:
    
    new_cooked_mrefs = [
        [__cook_sentence(ref, n, tokenizer) for ref in refs] for refs in mult_references
    ]
    new_cooked_cands = [__cook_sentence(cand, n, tokenizer) for cand in candidates]
    prev_cooked_cands += new_cooked_cands
    prev_cooked_mrefs += new_cooked_mrefs
    return prev_cooked_cands, prev_cooked_mrefs


def _cider_d_compute(
    cooked_cands: list[Counter],
    cooked_mrefs: list[list[Counter]],
    return_all_scores: bool,
    n: int,
    sigma: float,
    return_tfidf: bool,
    scale: float,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    if len(cooked_cands) <= 1:
        raise ValueError(
            f"CIDEr-D metric does not support less than 2 candidates with 2 references. (found {len(cooked_cands)} candidates, but expected > 1)"
        )
    # compute idf
    doc_frequencies = __compute_doc_freq(cooked_mrefs)
    # sanity check: assert to check document frequency
    assert len(cooked_cands) >= max(doc_frequencies.values()), "Sanity check failed."

    # compute log reference length
    log_n_refs = np.log(float(len(cooked_mrefs)))
    # compute cider score
    cider_d_scores, tfidf_lst = __compute_cider(
        cooked_cands,
        cooked_mrefs,
        doc_frequencies,
        log_n_refs,
        n,
        sigma,
        scale=scale,
    )
    cider_d_score = cider_d_scores.mean()

    cider_d_scores = torch.from_numpy(cider_d_scores)
    cider_d_score = torch.as_tensor(cider_d_score, dtype=torch.float64)

    if return_all_scores:
        cider_d_outs_corpus = {
            "cider_d": cider_d_score,
        }
        cider_d_outs_sents = {
            "cider_d": cider_d_scores,
        }
        if return_tfidf:
            cider_d_outs_sents["tfidf_lst"] = tfidf_lst  # type: ignore
        cider_d_outs = cider_d_outs_corpus, cider_d_outs_sents

        return cider_d_outs
    else:
        return cider_d_score


def __cook_sentence(
    sentence: str,
    n: int,
    tokenizer: Callable[[str], list[str]],
) -> Counter[tuple[str, ...]]:
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = tokenizer(sentence)
    counter = Counter()
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counter[ngram] += 1
    return counter


def __compute_doc_freq(cooked_mrefs: list[list[Counter]]) -> Counter[tuple[str, ...]]:
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    doc_frequencies = Counter()
    for refs in cooked_mrefs:
        all_refs_ngrams = set(ngram for ref in refs for ngram in ref.keys())
        for ngram in all_refs_ngrams:
            doc_frequencies[ngram] += 1

    return doc_frequencies


def __counter_to_vec(
    counters: dict[tuple, int],
    log_n_refs: float,
    n: int,
    doc_frequencies: Union[Mapping[tuple, int], Callable[[tuple], int]],
) -> tuple[list[defaultdict], np.ndarray, int]:
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: tf-idf of n-grams (array of n dict[tuple, float]), norm (array of n floats) (norms for n-grams), length (int) (number of distinct n-grams from 1 to n)
    """
    vec = [defaultdict(float) for _ in range(n)]
    length = 0
    norm = np.zeros((n,))

    for ngram, term_freq in counters.items():
        if isinstance(doc_frequencies, Mapping):
            count = doc_frequencies[ngram]
        else:
            count = doc_frequencies(ngram)

        # give ngram count 1 if it doesn't appear in reference corpus
        log_df = np.log(max(1.0, count))

        # ngram index
        cur_n = len(ngram) - 1

        # tf (term_freq) * idf (precomputed idf) for n-grams
        vec[cur_n][ngram] = float(term_freq) * (log_n_refs - log_df)

        # compute norm for the vector.  the norm will be used for computing similarity
        norm[cur_n] += pow(vec[cur_n][ngram], 2)

        if cur_n == 1:
            length += term_freq

    norm = np.sqrt(norm)
    return vec, norm, length


def __similarity(
    cand_vec: list[defaultdict],
    ref_vec: list[defaultdict],
    cand_norm: np.ndarray,
    ref_norm: np.ndarray,
    cand_len: int,
    ref_len: int,
    n: int,
    sigma: float,
) -> np.ndarray:
    """
    Compute the cosine similarity of two vectors.

    :param cand_vec: (n, nb_ngrams_of_len_n), contains the TFIDF scores for one candidate
    :param ref_vec: (n, nb_ngrams_of_len_n), contains the TFIDF scores for one reference
    :param cand_norm: (n,), norms of the candidate n-grams vectors
    :param ref_norm: (n,), norms of the reference n-grams vectors
    :param cand_len: Size of the candidate
    :returns: N-grams similarities as array of shape (n,)
    """
    delta = float(cand_len - ref_len)
    # measure consine similarity
    similarities = np.zeros((n,))

    for ni in range(n):
        # ngram
        for ngram, count in cand_vec[ni].items():
            # vrama91 : added clipping
            similarities[ni] += min(count, ref_vec[ni][ngram]) * ref_vec[ni][ngram]

        if (cand_norm[ni] != 0) and (ref_norm[ni] != 0):
            similarities[ni] /= cand_norm[ni] * ref_norm[ni]

        # vrama91: added a length based gaussian penalty
        similarities[ni] *= np.e ** (-(delta**2) / (2 * sigma**2))

    return similarities


def __compute_cider(
    cooked_cands: list[Counter],
    cooked_mrefs: list[list[Counter]],
    doc_frequencies: Union[Counter[tuple], Callable[[tuple], int]],
    log_n_refs: float,
    n: int,
    sigma: float,
    scale: float,
) -> tuple[np.ndarray, list[tuple[list, list]]]:
    scores = np.empty((len(cooked_cands),))
    tfidf_lst = []

    for i, (cand, refs) in enumerate(zip(cooked_cands, cooked_mrefs)):
        # compute vector for test captions
        vec, norm, length = __counter_to_vec(cand, log_n_refs, n, doc_frequencies)
        # compute vector for ref captions
        ngrams_scores = np.zeros((len(refs), n))
        vec_refs = []
        for j, ref in enumerate(refs):
            vec_ref, norm_ref, length_ref = __counter_to_vec(
                ref, log_n_refs, n, doc_frequencies
            )
            vec_refs.append(vec_ref)
            ngrams_scores[j] = __similarity(
                vec, vec_ref, norm, norm_ref, length, length_ref, n, sigma
            )

        # Use this weird mean calculation instead of ".mean()" because it can give slight differences compared to the original implementation
        score_avg = ngrams_scores.sum(axis=0).mean() / len(refs)
        scores[i] = score_avg
        tfidf_lst.append((vec, vec_refs))

    scores = scores * scale
    return scores, tfidf_lst

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def compute_spider(hyps, refs) -> float:
    refs = [[ref_] for ref_ in refs] # need to be List[List[str]]

    corpus_score, _ = cider_d(hyps, refs)
    
    spider = float(f"{corpus_score['cider_d'].item()*100.0:.4f}")
    print(f"SPIDEr: {spider}")