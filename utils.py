import pandas as pd
import re


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


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
    Save the measure-to-file mapping to a CSV file.
"""
def save_vocabulary_by_tag(V, output_dir):

    tag_files = {
        "M": "measures/measures.csv",
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


"""
    Save the file_code-measure to a CSV file.
"""
def save_code_measure(M, output_dir):

    measures_rows = []

    for measure, codes in M.items():
        for code in codes:
            measures_rows.append({"code": code, "measure": measure})

    measures_df = pd.DataFrame(measures_rows, columns=["code", "measure"])
    measures_df = measures_df.sort_values(["code", "measure"]).reset_index(drop=True)

    measures_df.to_csv(output_dir / "measures/code_measures.csv", index=False)

########################################################################################################################################################################################################################

########################################################################################################################################################################################################################
# Clustering utils functions
########################################################################################################################################################################################################################

"""
Remove .csv and whitespace from a code.
"""
def normalize_code(code):
    code = str(code).strip()
    return re.sub(r"\.csv$", "", code)


"""
Assign a domain to each code, according to its cluster
"""
def assign_domains_to_codes(cluster_domains, cluster_prefixes, output_path):

    cluster_prefixes = cluster_prefixes[["cluster", "code"]].copy()

    # Merge
    result = pd.merge(cluster_prefixes, cluster_domains, on="cluster", how="inner")
    result = result[["code", "domain"]]

    # Removes eventual duplicates
    result = result.drop_duplicates()

    # Save ordered result
    result = result.sort_values(["code", "domain"]).reset_index(drop=True)
    result.to_csv(output_path / "", index=False)

    return result