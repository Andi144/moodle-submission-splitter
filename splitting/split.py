import argparse
import math
import os.path
import re
import shutil
import zipfile
from collections.abc import Iterable, Sequence
from glob import glob

import numpy as np
import pandas as pd


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
    tutors.loc[dup, "name"] = [f"{dn} ({update_and_get_count(dn)})" for dn in dup_names]


def weighted_chunks(s: Sequence, weights: Iterable):
    # Scale weights to sum = 1.
    weights = np.array(weights, dtype=float) / sum(weights)
    chunk_sizes = [math.floor(len(s) * w) for w in weights]
    # Distribute the remaining elements evenly. Just repeatedly increase each
    # chunk size by 1 until we distributed all remaining elements.
    remainder_size = len(s) - sum(chunk_sizes)
    idx = 0
    while remainder_size > 0:
        chunk_sizes[idx] += 1
        idx = (idx + 1) % len(chunk_sizes)
        remainder_size -= 1
    assert sum(chunk_sizes) == len(s)
    # Chunk sizes are all set, now simply collect each chunk from "s".
    chunks = []
    idx = 0
    for chunk_size in chunk_sizes:
        chunks.append(s[idx:idx + chunk_size])
        idx += chunk_size
    assert sum([len(c) for c in chunks]) == len(s)
    return chunks


parser = argparse.ArgumentParser()
parser.add_argument("-sf", "--submissions_file", type=str, required=True,
                    help="Moodle ZIP file containing all submissions.")
parser.add_argument("-n", "--number", type=int,
                    help="Exercise number. If not specified, the Moodle ZIP filename is parsed to search for "
                         "various names (see '--exercise_names') and a following number (separated by arbitrary "
                         "whitespace characters, dashes or underscores). The number is used to roll the tutors, "
                         "i.e., to shift them across the created submission splits (e.g., if the tutors are "
                         "['A', 'B', 'C'] and the number is 1, then they are shifted by one, which results in "
                         "['C', 'A', 'B']). Use the value 0 if no such shift should be applied.")
parser.add_argument("--exercise_names", type=str, nargs="+",
                    default=["Assignment", "Exercise", "UE", "Übung", "Aufgabe"],
                    help="List of case-sensitive exercise names which will be used to automatically infer the "
                         "exercise number from the Moodle ZIP filename. The number is assumed to follow one "
                         "of these names, separated by arbitrary whitespace characters, dashes or underscores. "
                         "Default: ['Assignment', 'Exercise', 'UE', 'Übung', 'Aufgabe']")
tutors_group = parser.add_mutually_exclusive_group(required=True)
tutors_group.add_argument("-tf", "--tutors_file", type=str,
                          help="File containing tutor names (one name per line, no header). Optionally, a second "
                               "column (separated via a comma) can be provided that contains weights per tutor that "
                               "specify how the submission split sizes should be distributed. If no weights are "
                               "specified, then an equal split size distribution is assumed.")
tutors_group.add_argument("-tl", "--tutors_list", type=str, nargs="+",
                          help="List of tutor names. Optionally, a second entry (separated via a comma) can be "
                               "provided that contains weights per tutor that specify how the submission split sizes "
                               "should be distributed. If no weights are specified, then an equal split size "
                               "distribution is assumed.")
args = parser.parse_args()

# If the number of the exercise is specified, use it. Otherwise, try to extract/infer it from the submission filename.
exercise_num = args.number if args.number is not None else \
    extract_exercise_number(args.submissions_file, args.exercise_names)

tutors = pd.read_csv(args.tutors_file, header=None) if args.tutors_file is not None else \
    extract_weighted_tutors(args.tutors_list)
assert len(tutors.columns) == 1 or len(tutors.columns) == 2
# Assign equal default weights if only tutor names were specified to ensure we have a weight column.
if len(tutors.columns) == 1:
    tutors[1] = 1
tutors.columns = ["name", "weight"]
tutors["name"] = np.roll(tutors["name"], exercise_num)
tutors["weight"] = np.roll(tutors["weight"], exercise_num)
# Handle duplicate tutor names by simply adding increasing numbers after the name.
handle_duplicate_names(tutors)

unzip_dir = args.submissions_file + "_UNZIPPED"
print(f"extracting submissions ZIP file to '{unzip_dir}'")
with zipfile.ZipFile(args.submissions_file, "r") as f:
    f.extractall(unzip_dir)
submission_dirs = sorted(os.listdir(unzip_dir))

print(f"distributing {len(submission_dirs)} submissions among the following {len(tutors)} tutors:")
print(tutors)
for i, chunk in enumerate(weighted_chunks(submission_dirs, tutors["weight"])):
    chunk_file = f"{args.submissions_file[:-4]}_{tutors['name'][i]}.zip"
    with zipfile.ZipFile(chunk_file, "w") as f:
        # Write all files from the submission directory to the tutors ZIP file. Must exclude directories,
        # since glob includes them. Also specify the relative path as name in the ZIP file (arcname), as
        # otherwise, the full absolute path would be stored in the ZIP file.
        for submission_dir in chunk:
            for file in glob(os.path.join(unzip_dir, submission_dir, "**"), recursive=True):
                if os.path.isfile(file):
                    f.write(file, arcname=file[len(unzip_dir) + 1:])
    print(f"[{i + 1}/{len(tutors)}] {len(chunk):3d} submissions ---> {chunk_file}")

print(f"deleting extracted submissions directory '{unzip_dir}'")
shutil.rmtree(unzip_dir, ignore_errors=True)
