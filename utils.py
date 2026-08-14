from pathlib import Path
import pandas as pd
import re


########################################################################################################################################################################################################################
# Vocabulary construction utils functions (Phase I-II)
########################################################################################################################################################################################################################

"""
Extracts temporal and string attributes from a list of attributes.

An attribute containing a four-digit year between 1900 and 2100
is considered a candidate for being a temporal attribute. Once a
candidate is found, it is kept as long as all subsequent attributes
are temporal. If a non-temporal attribute is encountered, the
candidate is reset to None.

Once the first temporal attribute is identified, all preceding
attributes are considered string attributes, while all attributes
from the candidate onwards are considered temporal attributes.
"""
def split_attributes(attributes):

    candidate = None

    for i, attribute in enumerate(attributes):
        attribute = str(attribute)

        years = re.findall(r'\b\d{4}\b', attribute)

        is_temporal = any(1900 <= int(year) <= 2100 for year in years)

        if is_temporal:
            if candidate is None:
                candidate = i
        else:
            candidate = None

    if candidate is not None:
        return set(attributes[:candidate]), set(attributes[candidate:])

    return set(attributes), set()



"""
Parses a table title by identifying geographic and temporal terms,
and returns the remaining title without those terms.
"""
def parse_title(title, geo_vocab):
    words = title.split()

    geo = set()
    temporal = set()
    remaining_words = []

    for word in words:
        clean_word = word.strip(".,;:()[]{}")

        # Geographic value
        if clean_word in geo_vocab:
            geo.add(clean_word)
            continue

        # Year
        years = re.findall(r'\b\d{4}\b', clean_word)

        if any(1900 <= int(year) <= 2100 for year in years):
            temporal.update(years)
            continue

        remaining_words.append(word)

    remainder = " ".join(remaining_words)

    return remainder





def save_vocabulary_by_tag(V, output_directory):

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    terms_by_tag = {}

    for term, tags in V.items():
        for tag in tags:
            if tag not in terms_by_tag:
                terms_by_tag[tag] = set()

            terms_by_tag[tag].add(term)

    for tag, terms in terms_by_tag.items():
        df = pd.DataFrame(sorted(terms), columns=["term"])
        df.to_csv(output_directory / f"{tag}.csv", index=False)

########################################################################################################################################################################################################################