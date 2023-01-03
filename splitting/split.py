import argparse
import os.path
import shutil
import zipfile
from glob import glob

import numpy as np
import pandas as pd

from util import extract_exercise_number, extract_weighted_tutors, handle_duplicate_names, get_file_path, \
    get_submissions_df, match_full_names, weighted_chunks

FULL_NAME_COL = "full_name"
MOODLE_ID_COL = "moodle_id"
SUBMISSION_COL = "submission_file"

# TODO: encoding for every read and write access

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
parser.add_argument("--print_abs_paths", action="store_true",
                    help="If specified, the printed output will show the absolute paths of all files. Otherwise, only "
                         "the base filenames will be printed (default).")
parser.add_argument("--create_overview_file", action="store_true",
                    help="If specified, an overview CSV file will be created that contains all information on the "
                         "individual submissions and how they were distributed to the different tutors. This is useful "
                         "to quickly check which tutor corrected which submission. The name of the overview file will "
                         "be the same as '--submissions_file' but with the extension replaced with '.csv'.")
parser.add_argument("--sorting_keys", type=str, nargs="+", default=[],
                    help="If specified, the submissions will be sorted according to these keys. The keys must be part "
                         "of the header entries in the '--info_file', so this argument must be specified in addition. "
                         "By default, the submissions will simply be sorted according to their names.")
parser.add_argument("--submission_renaming_keys", type=str, nargs="+", default=[],
                    help="If specified, the submissions will be renamed according to these keys. The renaming will be "
                         "as follows: <key_1><sep><key_2>...<sep><key_n>, where <key_i> is the i-th key of these "
                         "specified renaming keys and <sep> is the separator as defined by argument "
                         "'--submission_renaming_separator'. The keys must be part of the header entries in the "
                         "'--info_file', so this argument must be specified in addition. Note that these keys "
                         "should lead to unique values for each submission, since otherwise, duplicate submission "
                         "names would be created, which leads to overwriting and hence loss of data. By default, the "
                         "original submission names will be used.")
parser.add_argument("--submission_renaming_separator", type=str, default="_",
                    help="The separator to use when renaming submissions (see '--submission_renaming_keys'). Has no "
                         "effect in case '--submission_renaming_keys' is not set. Default: '_'")
parser.add_argument("--info_file", type=str,
                    help="If specified, this must be a CSV file containing student (meta-)data that can be used for "
                         "sorting and renaming submissions. The CSV file must include a header, and these exact header "
                         "entries can be used as sort/renaming keys as defined by arguments '--sorting_keys' and "
                         "'--submission_renaming_keys', respectively. Typically, this is the course participants file "
                         "that can be downloaded in the participants overview in Moodle.")
parser.add_argument("--info_file_first_name_key", type=str,
                    help="Only relevant if '--sorting_keys' is specified. If so, this argument indicates the header "
                         "entry in the '--info_file' which contains the first name. Must be specified together with "
                         "argument '--info_file_last_name_key'. If not specified, then the first name header entry is "
                         "tried to be identified automatically based on string matching.")
parser.add_argument("--info_file_last_name_key", type=str,
                    help="Only relevant if '--sorting_keys' is specified. If so, this argument indicates the header "
                         "entry in the '--info_file' which contains the last name. Must be specified together with "
                         "argument '--info_file_first_name_key'. If not specified, then the last name header entry is "
                         "tried to be identified automatically based on string matching.")
args = parser.parse_args()

if args.sorting_keys and args.info_file is None:
    raise ValueError("must also specify '--info_file' when specifying '--sorting_keys'")
if args.submission_renaming_keys and args.info_file is None:
    raise ValueError("must also specify '--info_file' when specifying '--submission_renaming_keys'")
if args.sorting_keys and bool(args.info_file_first_name_key) != bool(args.info_file_last_name_key):
    raise ValueError("must specify either both or none of '--info_file_first_name_key' and 'info_file_last_name_key'")
if args.info_file:
    info_df = pd.read_csv(args.info_file)
    
    
    def check_keys(keys, arg_name):
        for k in keys:
            if k not in info_df.columns:
                raise ValueError(f"'--{arg_name}' contains '{k}' which is not part of the header entries in "
                                 f"'--info_file': {info_df.columns.to_list()}")
    
    
    if args.sorting_keys:
        check_keys(args.sorting_keys, "sorting_keys")
        if args.info_file_first_name_key:
            check_keys([args.info_file_first_name_key], "info_file_first_name_key")
        if args.info_file_last_name_key:
            check_keys([args.info_file_last_name_key], "info_file_last_name_key")
    if args.submission_renaming_keys:
        check_keys(args.submission_renaming_keys, "submission_renaming_keys")
        renaming_info_df = info_df[args.submission_renaming_keys]
        duplicates = renaming_info_df[renaming_info_df.duplicated()].drop_duplicates().reset_index(drop=True)
        if len(duplicates) > 0:
            raise ValueError(f"'--submission_renaming_keys' ({args.submission_renaming_keys}) leads to the following "
                             f"duplicate renaming values:\n{duplicates}\nMust choose different renaming keys")
else:
    info_df = None

# If the number of the exercise is specified, use it. Otherwise, try to extract/infer it from the submission filename.
exercise_num = args.number if args.number is not None else \
    extract_exercise_number(args.submissions_file, args.exercise_names)

tutors_df = pd.read_csv(args.tutors_file, header=None) if args.tutors_file is not None else \
    extract_weighted_tutors(args.tutors_list)
assert len(tutors_df.columns) == 1 or len(tutors_df.columns) == 2
# Assign equal default weights if only tutor names were specified to ensure we have a weight column.
if len(tutors_df.columns) == 1:
    tutors_df[1] = 1
tutors_df.columns = ["name", "weight"]
tutors_df["name"] = np.roll(tutors_df["name"], exercise_num)
tutors_df["weight"] = np.roll(tutors_df["weight"], exercise_num)
# Handle duplicate tutor names by simply adding increasing numbers after the name.
handle_duplicate_names(tutors_df)

unzip_dir = args.submissions_file + "_UNZIPPED"
print(f"extracting submissions ZIP file to '{get_file_path(unzip_dir, args.print_abs_paths)}'")
with zipfile.ZipFile(args.submissions_file, "r") as f:
    f.extractall(unzip_dir)
# To extract data, the following format is assumed for each submission (correct at the time of writing this code):
# <full student name>_<7-digit moodle ID>_<rest of submission string>
# where <full student name> is a space-separated list of strings that holds the full student name, i.e., all first
# names and all last names (however, we do not know which parts belong to first names and which to last names),
# <7-digit moodle ID> is an ID with 7 digits generated by Moodle, and <rest of submission string> can be an arbitrary
# string (at the time of writing this code, this is the string "assignsubmission_file_").
# TODO: create arguments for all these columns and regex patterns in case the Moodle format changes (currently, this
#  would require code modification right here)
submissions_df = get_submissions_df(os.listdir(unzip_dir), regex_cols={
    FULL_NAME_COL: r".+(?=_\d{7})",  # Extract the full name according to the above format.
    MOODLE_ID_COL: r"\d{7}",  # Extract the 7-digit Moodle ID according to the above format.
    SUBMISSION_COL: r".+",  # This is simply the entire submission (no specific extraction of a pattern).
})
if args.sorting_keys:
    first_name_col = args.info_file_first_name_key
    last_name_col = args.info_file_last_name_key
    if first_name_col is None:
        first_name_col, last_name_col = match_full_names(submissions_df[FULL_NAME_COL], info_df)
        print(f"identified '{first_name_col}' as first name column and '{last_name_col}' as last name column")
    info_df[FULL_NAME_COL] = info_df[first_name_col] + " " + info_df[last_name_col]
    merged_df = pd.merge(submissions_df, info_df, on=FULL_NAME_COL, how="inner")
    assert len(submissions_df) == len(merged_df)
    print(f"sorting submissions according to: {', '.join(args.sorting_keys)}")
    submissions_df = merged_df.sort_values(by=args.sorting_keys)
else:
    submissions_df.sort_values(SUBMISSION_COL, inplace=True)

if args.submission_renaming_keys:
    name_format = args.submission_renaming_separator.join(f"<{k}>" for k in args.submission_renaming_keys)
    print(f"renaming submissions according to the following format: {name_format}")

if args.create_overview_file:
    overview_file = os.path.splitext(args.submissions_file)[0] + ".csv"
    print(f"storing overview file to '{get_file_path(overview_file, args.print_abs_paths)}'")
else:
    overview_file = None

print(f"distributing {len(submissions_df)} submissions among the following {len(tutors_df)} tutors:")
print(tutors_df)
for i, chunk_df in enumerate(weighted_chunks(submissions_df, tutors_df["weight"])):
    if args.create_overview_file:
        assert overview_file is not None
        chunk_df[["tutor_name", "tutor_weight"]] = tutors_df[["name", "weight"]].iloc[i]
        # The first chunk (i == 0) is handled differently: First, the file will be newly created (mode "w"). Second, the
        # header will be written. In all following cases (i >= 1), submissions will simply be appended (mode "a") and no
        # header will be written anymore (not needed since it already exists because of the first chunk at i == 0).
        first_chunk = i == 0
        chunk_df.to_csv(overview_file, mode="w" if first_chunk else "a", header=first_chunk, index=False)
    
    chunk_file = f"{args.submissions_file[:-4]}_{tutors_df['name'][i]}.zip"
    with zipfile.ZipFile(chunk_file, "w") as f:
        # Write all files from the submission directory to the tutors ZIP file. Must exclude directories, since glob
        # includes them. Also specify the relative path as name in the ZIP file (arcname), as otherwise, the full
        # absolute path would be stored in the ZIP file.
        for _, entry in chunk_df.iterrows():
            for file in glob(os.path.join(unzip_dir, entry[SUBMISSION_COL], "**"), recursive=True):
                if os.path.isfile(file):
                    if args.submission_renaming_keys:
                        name = args.submission_renaming_separator.join(entry[k] for k in args.submission_renaming_keys)
                        name = os.path.join(name, os.path.basename(file))
                    else:
                        name = file[len(unzip_dir) + 1:]
                    f.write(file, arcname=name)
    print(f"[{i + 1}/{len(tutors_df)}] {len(chunk_df):3d} submissions ---> "
          f"{get_file_path(chunk_file, args.print_abs_paths)}")

print(f"deleting extracted submissions directory '{get_file_path(unzip_dir, args.print_abs_paths)}'")
shutil.rmtree(unzip_dir, ignore_errors=True)
