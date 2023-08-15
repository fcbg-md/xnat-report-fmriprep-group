
from quality import get_bids_data, generate_figure, generate_report_with_plots, generate_figure2, afficher_aseg

from quality import get_bids_data, generate_figure, generate_report_with_plots, generate_figure2, bids_subjects

import uuid
import os
import pandas as pd
from pathlib import Path


data_path = "/Users/attiaaxel/Desktop/bids_fmriprep_1subj/"
output_dir=os.path.join(data_path, "report")
#output_dir = os.path.join(data_path, "report")
reportlets_dir = os.path.join(output_dir, "report/reportlets/figures")
all_tables, entities, repetition_times = get_bids_data(data_path)
print(os.path.exists(reportlets_dir))

subject = "sub-s05"
afficher_aseg(data_path, subject)


generate_figure(all_tables, repetition_times, 'global_signal', output_dir)
generate_figure(all_tables,repetition_times, 'csf', output_dir)
generate_figure(all_tables,repetition_times, 'white_matter', output_dir)
generate_figure2(all_tables, repetition_times, ['rot_x', 'rot_y', 'rot_z'], output_dir)
generate_figure2(all_tables, repetition_times, ['trans_x', 'trans_y', 'trans_z'], output_dir)


generate_figure(all_tables,repetition_times, 'framewise_displacement', output_dir)
generate_figure(all_tables,repetition_times, 'dvars', output_dir)
generate_figure(all_tables,repetition_times, 'std_dvars', output_dir)
generate_figure(all_tables,repetition_times, 'rmsd', output_dir)

#perform_pca(all_tables, output_dir)
#display_motion_outliers(all_tables)

data_path = "/home/axel/Test2/bids_fmriprep_1subj"
output_dir =  os.path.join(data_path)
sub_id_value = bids_subjects(data_path)
#output_dir = os.path.join(data_path, "report")
reportlets_dir = Path(output_dir) / "report" / "reportlets" / sub_id_value  / "figures"
reportlets_dir.mkdir(parents=True, exist_ok=True)
all_tables, repetition_times = get_bids_data(data_path)

generate_figure(all_tables, repetition_times, 'global_signal', reportlets_dir)
generate_figure(all_tables, repetition_times, 'csf', reportlets_dir)
generate_figure(all_tables, repetition_times, 'white_matter', reportlets_dir)
generate_figure2(all_tables, repetition_times, ['rot_x', 'rot_y', 'rot_z'], reportlets_dir)
generate_figure2(all_tables, repetition_times, ['trans_x', 'trans_y', 'trans_z'], reportlets_dir)


generate_figure(all_tables, repetition_times, 'framewise_displacement', reportlets_dir)
generate_figure(all_tables, repetition_times, 'dvars', reportlets_dir)
generate_figure(all_tables, repetition_times, 'std_dvars', reportlets_dir)
generate_figure(all_tables, repetition_times, 'rmsd', reportlets_dir)
generate_figure(all_tables, repetition_times, 'cosine00', reportlets_dir)



bootstrap_file="./container/bootstrap-fmriprep.yml"
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

report_dir=Path(output_dir) / "report"

report_filename = generate_report_with_plots(
    output_dir=report_dir,
    run_uuid=uuid.uuid4(), 
    reportlets_dir=reportlets_dir,
    bootstrap_file=bootstrap_file,
    metadata=metadata,
    plugin_meta={}
)



