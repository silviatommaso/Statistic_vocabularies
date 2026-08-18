import pandas as pd
import os

from utils.steps_I_VI.vocabulary_construction import parse_title


"""
    Parse the Eurostat table of contents file.

    The function reads the TSV file, extracts the original indentation
    of each title, removes unnecessary whitespace, and returns a
    cleaned DataFrame.
"""
def parse_toc(filepath):

    df = pd.read_csv(filepath, sep="\t", dtype=str)

    # Store the original indentation before removing it
    def get_indent(title):
        if pd.isna(title):
            return 0

        return len(title) - len(title.lstrip(" "))

    df["indent"] = df["title"].apply(get_indent)

    # Remove indentation from titles
    df["title"] = df["title"].str.strip()

    # Clean whitespace in other columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df



"""
    Build hierarchical breadcrumbs for the entries in the TOC.

    The hierarchy is reconstructed using the indentation of each entry.
    Table titles are parsed to remove geographic information before
    being included in the breadcrumb.
"""
def build_breadcrumbs(df):

    breadcrumbs = []
    stack = []

    for _, row in df.iterrows():

        title = row["title"]
        indent = row["indent"]

        # Remove elements that are at the same or deeper level
        # than the current entry.
        while stack and stack[-1][0] >= indent:
            stack.pop()

        # The remaining stack contains the parents
        breadcrumb_parts = [item[1] for item in stack]

        # Add the current entry
        breadcrumb_parts.append(title)
        breadcrumb = " > ".join(breadcrumb_parts)
        breadcrumbs.append(breadcrumb)

        # Add current entry to the hierarchy
        stack.append((indent, title))

    result = df[["code", "type"]].copy()
    result["breadcrumb"] = breadcrumbs

    return result



"""
    Filter the breadcrumb DataFrame by matching table codes with
    the names of the CSV files in the tables directory.
"""
def filter_breadcrumbs_by_tables(toc_file_path, tables_dir):
    
    table_codes = {os.path.splitext(filename)[0] for filename in os.listdir(tables_dir) if filename.lower().endswith(".csv")}

    # get breadcumbs of files in TOC
    breadcrumbs = build_breadcrumbs(parse_toc(toc_file_path))
    breadcrumb_codes = set(breadcrumbs["code"])

    filtered_rows = []
    matched_codes = set()

    # Match each table file with the corresponding TOC entry
    for table_code in table_codes:

        # Normal case: exact match
        if table_code in breadcrumb_codes:
            row = breadcrumbs[breadcrumbs["code"] == table_code].iloc[0].copy()
            filtered_rows.append(row)
            matched_codes.add(table_code)

        # Special case: codes containing '$'
        elif "$" in table_code:
            base_code = table_code.split("$", 1)[0]

            if base_code in breadcrumb_codes:
                row = breadcrumbs[breadcrumbs["code"] == base_code].iloc[0].copy()

                # Keep the original table code
                row["code"] = table_code

                filtered_rows.append(row)
                matched_codes.add(table_code)

    # Build filtered DataFrame
    filtered = pd.DataFrame(filtered_rows, columns=["code", "type", "breadcrumb"])
    filtered.reset_index(drop=True, inplace=True)

    # Find table codes that do not have a matching breadcrumb
    unmatched_codes = sorted(table_codes - matched_codes)

    # Save unmatched codes
    pd.DataFrame({"code": unmatched_codes}).to_csv("input/auxiliar_files/TOC/unmatched_codes.csv", index=False)

    # Save filtered breadcrumbs
    filtered.to_csv("input/auxiliar_files/TOC/breadcrumbs_filtered.csv", index=False)

#######################################################################################################################################################################################################################################################


if __name__ == "__main__":

    filter_breadcrumbs_by_tables(
        toc_file_path="input/auxiliar_files/TOC/table_of_contents_en.txt",
        tables_dir="input/tables/eurostat_7605_tables"
    )