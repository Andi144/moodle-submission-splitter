import math
import os.path
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


# TODO: handle empty tutors and submissions


def extract_exercise_number(submissions_file: str, exercise_names: Iterable[str]):
    for ex_name in exercise_names:
        match = re.search(rf"{ex_name}[\s\-_]*(\d+)", os.path.basename(submissions_file))
        if match:
            return int(match.group(1))
    raise ValueError("could not automatically infer exercise number, must specify manually via '-n'")


def extract_weighted_tutors(tutors_list: Sequence[str]):
    # Quick check to determine whether weights are specified.
    if "," in tutors_list[0]:
        rows = []
        for t in tutors_list:
            if "," not in t:
                raise ValueError(f"expected ',' in tutor entry '{t}'")
            name, weight = t.split(",", maxsplit=1)
            rows.append([name, float(weight)])
        return pd.DataFrame(rows)
    for t in tutors_list:
        if "," in t:
            raise ValueError(f"unexpected ',' in tutor entry '{t}'")
    return pd.DataFrame(tutors_list)


def handle_duplicate_names(tutors_df: pd.DataFrame):
    dup = tutors_df["name"].duplicated(keep=False)
    dup_names = tutors_df["name"][dup]
    # Create a count for each unique tutor name.
    counts = dict()
    
    def update_and_get_count(name: str):
        count = counts.get(name, 0)
        count += 1
        counts[name] = count
        return count
    
    # Change DataFrame inplace.
    tutors_df.loc[dup, "name"] = [f"{dn} ({update_and_get_count(dn)})" for dn in dup_names]


def get_submissions_df(submissions: Iterable[str], regex_cols: dict[str, str]):
    data = defaultdict(list)
    for s in submissions:
        for name, regex in regex_cols.items():
            match = re.search(regex, s)
            if match is None:
                raise ValueError(f"submission '{s}' does not contain regex pattern '{regex}' for column '{name}'")
            data[name].append(match.group())
    return pd.DataFrame(data)


def match_full_names(full_names: pd.Series, info_df: pd.DataFrame):
    # Try to match the full names (given in the submissions) to separate first and last names. This is a bit tricky,
    # since a full name is just a space-separated string that starts with the first name and ends with the last name,
    # but both the first name and the last name might be multi-names, and there is no way of knowing to which a single
    # name element belongs. So we must find out by trying to match the full names to individual first and last names.
    first_name_counts = defaultdict(int)
    last_name_counts = defaultdict(int)
    
    for full_name in full_names:
        for col in info_df.columns:
            for elem in info_df[col]:
                # Rather simple check for first names (this relies on the fact that the full names are a combination of
                # first names followed by last names, and not the other way around).
                if full_name.startswith(elem):
                    first_name_counts[col] += 1
                else:
                    # We end up here if it is most likely not a first name. Might result in some false negatives, but
                    # that's ok. The name part at index 0 is always part of the first name (again, this relies on the
                    # fact that the full names are a combination of first names followed by last names, and not the
                    # other way around), so we can immediately drop it here.
                    name_parts = full_name.split(" ")
                    assert len(name_parts) >= 2
                    
                    for name_part in name_parts[1:]:
                        # Most likely, we will get some last name counts for the first name column (it is not uncommon
                        # that a single name part also appears as first name), but overall, the actual last name column
                        # count should be higher in the end.
                        if elem.startswith(name_part):
                            last_name_counts[col] += 1
    
    # Heuristic: Check counts to see which columns contain first/last names.
    first_name_col = max(first_name_counts.items(), key=lambda key_val: key_val[1])[0]
    last_name_col = max(last_name_counts.items(), key=lambda key_val: key_val[1])[0]
    assert first_name_col != last_name_col
    return first_name_col, last_name_col


def weighted_chunks(df: pd.DataFrame, weights: Iterable):
    # Scale weights to sum = 1.
    weights = np.array(weights, dtype=float) / sum(weights)
    chunk_sizes = [math.floor(len(df) * w) for w in weights]
    # Distribute the remaining elements evenly. Just repeatedly increase each chunk size by 1 until we distributed all
    # remaining elements.
    remainder_size = len(df) - sum(chunk_sizes)
    idx = 0
    while remainder_size > 0:
        chunk_sizes[idx] += 1
        idx = (idx + 1) % len(chunk_sizes)
        remainder_size -= 1
    assert sum(chunk_sizes) == len(df)
    # Chunk sizes are all set, now simply collect each chunk from "df".
    chunks = []
    idx = 0
    for chunk_size in chunk_sizes:
        chunks.append(df.iloc[idx:idx + chunk_size].copy())
        idx += chunk_size
    assert sum([len(c) for c in chunks]) == len(df)
    return chunks


def get_file_path(path: str, absolute: bool):
    return os.path.abspath(path) if absolute else os.path.basename(path)
