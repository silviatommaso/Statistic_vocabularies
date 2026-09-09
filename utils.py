import pandas as pd
import re
from pathlib import Path


########################################################################################################################################################################################################################
# Vocabulary construction utils functions
########################################################################################################################################################################################################################

"""
Discover whether an attribute is numerical or not
"""
def is_numeric(value):
    if not isinstance(value, str):
        return False

    value = value.strip().replace(",", ".")

    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z])?", value))


"""
Discover whether an attribute is a date or not
"""
def is_date(value):

    YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}$")
    DATE_DMY_PATTERN = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}$")
    DATE_YMD_PATTERN = re.compile(r"^(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}$")

    if not isinstance(value, str):
        return False

    value = value.strip()

    # Single year: 2022
    if YEAR_PATTERN.fullmatch(value):
        return True

    # Single date: 20-12-2022 or 20/12/2022
    if DATE_DMY_PATTERN.fullmatch(value):
        return True

    # Single date: 2022-12-20 or 2022/12/20
    if DATE_YMD_PATTERN.fullmatch(value):
        return True

    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
    

TIME_MARKER_PATTERN = re.compile(r"\\\s*(?:time|time_period)\s*$", re.IGNORECASE)


"""
Returns True if the attribute name ends with a temporal marker
(e.g. "geo\\TIME_PERIOD", "geo\\time"), used both to find the
string/temporal split point and to strip the marker for display.
"""
def has_time_marker(attribute):
    return bool(TIME_MARKER_PATTERN.search(str(attribute)))


"""
Strips the temporal marker from an attribute name, if present
(e.g. "geo\\TIME_PERIOD" -> "geo"). Leaves other attributes unchanged.
"""
def strip_time_marker(attribute):
    return TIME_MARKER_PATTERN.sub("", str(attribute))



def split_attributes(attributes):

    candidate = None

    for i, attribute in enumerate(attributes):
        if has_time_marker(attribute):
            candidate = i + 1
            break

    if candidate is not None:
        return set(attributes[:candidate]), set(attributes[candidate:])

    return set(attributes), set()



"""
Parses a table title by identifying geographic and temporal terms,
and returns the remaining title without those terms.

Geographic matching is done on WORD SEQUENCES (n-grams), not single
words: geo_vocab (the NUTS dictionary) contains many multi-word entity
names (e.g. "North Macedonia", "United Kingdom", "Bosnia and
Herzegovina"). None of their individual words are geo terms on their
own, so splitting the title into single words and checking each one in
isolation -- the previous approach -- silently misses every multi-word
entity. Instead, at each position we try the longest possible phrase
first (greedy longest match) and fall back to shorter phrases, so e.g.
"North Macedonia" is matched as one unit rather than missed entirely.
"""
def parse_title(title, geo_vocab, max_geo_words=6):

    words = title.split()
    n = len(words)

    geo = set()
    temporal = set()
    remaining_words = []

    i = 0
    while i < n:

        matched_span = None

        # Try progressively shorter phrases starting at i, longest first,
        # so multi-word geo entities are preferred over any partial match.
        max_span = min(max_geo_words, n - i)
        for span in range(max_span, 0, -1):
            phrase_words = words[i:i + span]
            # only strip trailing punctuation from the phrase's last word
            phrase = " ".join(phrase_words[:-1] + [phrase_words[-1].strip(".,;:()[]{}")])

            if phrase in geo_vocab:
                geo.add(phrase)
                matched_span = span
                break

        if matched_span is not None:
            i += matched_span
            continue

        word = words[i]
        clean_word = word.strip(".,;:()[]{}")

        # Year
        years = re.findall(r'\b\d{4}\b', clean_word)

        if any(1900 <= int(year) <= 2100 for year in years):
            temporal.update(years)
            i += 1
            continue

        remaining_words.append(word)
        i += 1

    remainder = " ".join(remaining_words)

    return remainder


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
    Save the measure-to-file mapping to a CSV file.
"""
def save_vocabulary_by_tag(V, output_dir):

    tag_files = {
        "M": "measures.csv",
        "N": "dimension_names.csv",
        "A": "dimension_values.csv",
        "U": "units.csv"
    }

    for tag, filename in tag_files.items():

        terms = []

        for term, tags in V.items():
            if tag in tags:
                terms.append(term)

        df = pd.DataFrame({"term": sorted(terms)})

        df.to_csv(output_dir / filename, index=False)