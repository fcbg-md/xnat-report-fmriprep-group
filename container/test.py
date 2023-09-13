#!/usr/bin/env python3
import time

start_time = time.time()

from quality import extract_unique_tasks, display_outliers
from quality import get_bids_data, generate_report_with_plots, generate_figure, generate_figures_motion
import uuid
import os
from pathlib import Path
import argparse
 
#data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
#output_dir="/home/axel/Test2/bids_fmriprep_10subj/report"

#data_path = "/home/axel/Test2/bids_fmriprep_1subj/"
#output_dir=os.path.join(data_path, "report")

# parser = argparse.ArgumentParser(description='Process data_path for the script.')
# parser.add_argument('data_path', type=str, help='Path to the data')
# args = parser.parse_args()

#data_path = args.data_path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process data_path for the script.')
    parser.add_argument('data_path', type=str, help='Path to the data')
    parser.add_argument('--fd_threshold', type=float, default=0.5, help='Seuil de FD')
    parser.add_argument('--dvars_threshold', type=float, default=1.5, help='Seuil de DVARS')

    args = parser.parse_args()

    data_path = args.data_path
    fd_threshold = args.fd_threshold
    dvars_threshold = args.dvars_threshold
    output_dir=os.path.join(data_path, "report")


reportlets_dir = Path(output_dir) / "reportlets" / "figures"
reportlets_dir.mkdir(parents=True, exist_ok=True)
all_tables, entities, repetition_times = get_bids_data(data_path)
print(os.path.exists(reportlets_dir))

tasks = extract_unique_tasks(all_tables)

all_tasks = []


display_outliers(all_tables, repetition_times, reportlets_dir, 3, 5)

generate_figure(all_tables, repetition_times, 'global_signal', reportlets_dir)
generate_figure(all_tables, repetition_times, 'csf', reportlets_dir)
generate_figure(all_tables, repetition_times, 'white_matter', reportlets_dir)
generate_figures_motion(all_tables, repetition_times, ['rot_x', 'rot_y', 'rot_z'], reportlets_dir)
generate_figures_motion(all_tables, repetition_times, ['trans_x', 'trans_y', 'trans_z'], reportlets_dir)
generate_figure(all_tables, repetition_times, 'framewise_displacement', reportlets_dir)    
generate_figure(all_tables, repetition_times, 'std_dvars', reportlets_dir)
generate_figure(all_tables, repetition_times, 'rmsd', reportlets_dir)
generate_figure(all_tables, repetition_times, 'dvars', reportlets_dir)



bootstrap_file="./bootstrap-fmriprep.yml"
prov = {}

metadata={
    "dataset": "Test dataset Freesurfer report",
    "fs-metadata": {
        "Provenance Information": prov  
    }
}


report_filename = generate_report_with_plots(
    output_dir=output_dir,
    run_uuid=uuid.uuid4(), 
    reportlets_dir=reportlets_dir,
    bootstrap_file=bootstrap_file,
    metadata=metadata,
    tasks=all_tasks,
    plugin_meta={}
)

print(f"Report generated in {report_filename}")


end_time = time.time()

print(f"running time {end_time - start_time} seconds.")

