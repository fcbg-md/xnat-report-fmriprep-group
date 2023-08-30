#!/usr/bin/env python3
import time

start_time = time.time()

from test2 import generate_figure2
from test2 import get_bids_data, generate_report_with_plots, generate_figure
import uuid
import os
import pandas as pd
from pathlib import Path
import argparse

data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
output_dir="/home/axel/Test2/bids_fmriprep_10subj/report"

#data_path = "/home/axel/Test2/bids_fmriprep_1subj/"
#output_dir=os.path.join(data_path, "report")

# parser = argparse.ArgumentParser(description='Process data_path for the script.')
# parser.add_argument('data_path', type=str, help='Path to the data')
# args = parser.parse_args()

#data_path = args.data_path


reportlets_dir = Path(output_dir) / "reportlets" / "figures"
reportlets_dir.mkdir(parents=True, exist_ok=True)
all_tables, entities, repetition_times = get_bids_data(data_path)
print(os.path.exists(reportlets_dir))

#tasks = extract_unique_tasks(all_tables)


all_tasks = []

#global_data, motion_outliers_list, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "global_signal")
all_tasks.extend(generate_figure(all_tables, repetition_times, 'global_signal', reportlets_dir))
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "csf")
# all_tasks.extend(generate_figure(all_tables, repetition_times, 'csf', reportlets_dir, motion_outliers_combined))
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "white_matter")
# all_tasks.extend(generate_figure(all_tables, repetition_times, 'white_matter', reportlets_dir, motion_outliers_combined))
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "a_comp_cor_00")
all_tasks.extend(generate_figure2(all_tables, repetition_times, ['rot_x', 'rot_y', 'rot_z'], reportlets_dir)) 
# all_tasks.extend(generate_figure2(all_tables, repetition_times, ['trans_x', 'trans_y', 'trans_z'], reportlets_dir)) 
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "framewise_displacement")
# all_tasks.extend(generate_figure(all_tables, repetition_times, 'framewise_displacement', reportlets_dir, motion_outliers_combined))
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "std_dvars")
# all_tasks.extend(generate_figure(all_tables, repetition_times, 'std_dvars', reportlets_dir, motion_outliers_combined))
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "rmsd")
# all_tasks.extend(generate_figure(all_tables, repetition_times, 'rmsd', reportlets_dir, motion_outliers_combined))
# global_data, motion_outliers_combined, repetition_time = read_and_preprocess_data(tasks, all_tables, repetition_times, "dvars")
# all_tasks.extend(generate_figure(all_tables, repetition_times, 'dvars', reportlets_dir, motion_outliers_combined))


#display_motion_outliers(all_tables)

#output_dir = os.path.join(data_path, "report")
#reportlets_dir = Path(output_dir) / "report" / "reportlets" / sub_id_value  / "figures"
#reportlets_dir.mkdir(parents=True, exist_ok=True)




bootstrap_file="./bootstrap-fmriprep.yml"
prov = {}

metadata={
    "dataset": "Test dataset Freesurfer report",
    "fs-metadata": {
        "Provenance Information": prov  
    }
}

# layout = BIDSLayout(in_folder)
# entities = {}
# files = layout.get()
# for file in files:
#     entities.update(file.get_entities())

report_dir=output_dir

report_filename = generate_report_with_plots(
    output_dir=report_dir,
    run_uuid=uuid.uuid4(), 
    reportlets_dir=reportlets_dir,
    bootstrap_file=bootstrap_file,
    metadata=metadata,
    tasks=all_tasks,
    plugin_meta={}
)


end_time = time.time()

print(f"running time {end_time - start_time} seconds.")

