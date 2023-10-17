import itertools
import math
import os.path
import re
from collections import namedtuple, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


# TODO: handle empty tutors and submissions


def extract_exercise_number(submissions_file: str, exercise_names: Iterable[str]) -> int:
    for ex_name in exercise_names:
        match = re.search(rf"{ex_name}[\s\-_]*(\d+)", os.path.basename(submissions_file))
        if match:
            return int(match.group(1))
    raise ValueError("could not automatically infer exercise number, must specify manually via '-n'")


def extract_weighted_tutors(tutors_list: Sequence[str]) -> pd.DataFrame:
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


def get_submissions_df(submissions: Iterable[str], regex_cols: dict[str, str]) -> pd.DataFrame:
    data = defaultdict(list)
    for s in submissions:
        for name, regex in regex_cols.items():
            match = re.search(regex, s)
            if match is None:
                raise ValueError(f"submission '{s}' does not contain regex pattern '{regex}' for column '{name}'")
            data[name].append(match.group())
    return pd.DataFrame(data)


def match_full_names(full_names: pd.Series, info_df: pd.DataFrame) -> tuple[str, str]:
    # Try to match the full names (given in the submissions) to separate first and last names. This is a bit tricky,
    # since a full name is just a space-separated string that starts with the first name and ends with the last name,
    # but both the first name and the last name might be multi-names, and there is no way of knowing to which a single
    # name element belongs. So we must find out by trying to match the full names to individual first and last names.
    # The idea here is to just try all possible 2-permutations of the info_df columns, chain the elements together with
    # a space, and then checking whether these chained elements are the same as the full names. If so, the first column
    # must be the one containing first names and the second column the one containing last names. Note: The more columns
    # the info_df has, the more inefficient this heuristic becomes because of the permutations (however, we need the
    # permutations since we do not know the order of the columns in info_df). With many columns, it is this highly
    # recommended to manually provide the first name and last name columns.
    Mismatch = namedtuple("Mismatch", ["col1", "col2", "df"])
    closest_mismatch = None
    for col1, col2 in itertools.permutations(info_df.columns, 2):
        full_names_candidates = info_df[col1] + " " + info_df[col2]
        matching = full_names.isin(full_names_candidates)
        if matching.all():
            return col1, col2
        mismatching = full_names[~matching]
        if closest_mismatch is None or len(mismatching) < len(closest_mismatch.df):
            closest_mismatch = Mismatch(col1, col2, mismatching)
    raise ValueError(f"could not identify first name and last name columns; closest mismatch for columns "
                     f"'{closest_mismatch.col1}' and '{closest_mismatch.col2}':\n{closest_mismatch.df}")


def weighted_chunks(df: pd.DataFrame, weights: Iterable) -> list[pd.DataFrame]:
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


def get_file_path(path: str, absolute: bool) -> str:
    return os.path.abspath(path) if absolute else os.path.basename(path)
