import argparse
import os

import pandas as pd


# TODO: lots of hard-coded assumptions (should be parameterized)
def get_missing_df(grading_file, tutors_overview_file):
    grading_df = pd.read_csv(grading_file, dtype=str)
    # Exclude the following entries:
    # 1) "Status" contains " - Bewertet" (was already graded)
    # 2) "Status" contains "Keine Abgabe" (cannot be graded; it might be that is does contain a grade due to
    #    extraordinary circumstances, e.g., late submission which was not done via Moodle, but we are only interested
    #    in actually submitted entries that are not graded yet, so we can skip all these entries here)
    # 3) "Bewertung kann geändert werden" equals "Nein" (was already graded, but the grading was done vis an alternative
    #    CSV upload in Moodle, which, oddly enough, does not change the "Status")
    missing_df = grading_df[~(grading_df["Status"].str.contains(" - Bewertet") |
                              grading_df["Status"].str.contains("Keine Abgabe") |
                              (grading_df["Bewertung kann geändert werden"] == "Nein"))]
    tutors_df = pd.read_csv(tutors_overview_file, dtype=str)
    merged_missing_df = pd.merge(missing_df, tutors_df, on="ID-Nummer")
    assert len(missing_df) == len(merged_missing_df), "there are students not assigned to tutors"
    return merged_missing_df


parser = argparse.ArgumentParser()
parser.add_argument("--grading_files", nargs="+", type=str, required=True,
                    help="List of Moodle grading CSV files, each obtained via 'Download grading worksheet' (German: "
                         "'Bewertungstabelle herunterladen'). The order of this list must exactly match the order of "
                         "'--tutors_overview_files'.")
parser.add_argument("--tutors_overview_files", nargs="+", type=str, required=True,
                    help="List of tutor overview CSV files, each obtained via the script 'split.py' with argument "
                         "'--create_overview_file' enabled. The order of this list must exactly match the order of "
                         "'--grading_files'.")
parser.add_argument("--print_missing", action="store_true",
                    help="If specified, all individual student submission where the feedback/grading is still missing "
                         "are printed to the console.")
parser.add_argument("--skip_sanity_check", action="store_true",
                    help="If specified, skips the sanity check which matches each file of '--grading_files' with each "
                         "file of '--tutors_overview_files'. This check is based on the default (Moodle) filenames, so "
                         "if the provided '--grading_files' and '--tutors_overview_files' differ from this default "
                         "naming format, then this check should be disabled or it will most likely fail.")
args = parser.parse_args()

if len(args.grading_files) != len(args.tutors_overview_files):
    raise ValueError("grading_files must match tutors_overview_files exactly")

if args.print_missing:
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100
    pd.options.display.expand_frame_repr = False

for gf, tof in zip(args.grading_files, args.tutors_overview_files):
    gf_basename = os.path.basename(gf)
    tof_basename = os.path.basename(tof)
    # Small sanity check if we are comparing matching files. This only works in case of default (Moodle) filenames
    if not args.skip_sanity_check:
        # Expected grading filename format: Bewertungen-<COURSE>-<ASSIGNMENT>-<FIXED_ID>.csv
        # Expected tutors overview filename format:     <COURSE>-<ASSIGNMENT>-<FIXED_ID>.csv
        if gf_basename[-len(tof_basename):] != tof_basename:
            raise ValueError(f"failed sanity check: mismatching files:\n-> {gf_basename}\n-> {tof_basename}")
    
    print(f"Grading file:  {gf_basename}")
    print(f"Overview file: {tof_basename}")
    print("-" * 75)
    mdf = get_missing_df(gf, tof)
    tutor_groups = mdf.groupby("tutor_name")
    for i, (tutor, group_df) in enumerate(tutor_groups):
        print(f"[{i + 1}/{len(tutor_groups)}] {tutor}: {len(group_df)} missing assignment "
              f"feedback{'' if len(group_df) == 1 else 's'}")
        if args.print_missing:
            # TODO: hard-coded column names
            print_df = group_df.reset_index(drop=True)
            if "Vorname" in group_df and "Nachname" in group_df and "ID-Nummer" in group_df:
                print_df.rename(columns={"Vorname": "first_name", "Nachname": "last_name", "ID-Nummer": "id"},
                                inplace=True)
                print_df = print_df[["first_name", "last_name", "id"]]
                print(print_df)
            else:
                print(print_df)
    print("-" * 75)
