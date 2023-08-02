from quality import get_bids_data, generate_figure, generate_report_with_plots
import uuid
import os

data_path = "./bids_fmriprep_1subj"
output_dir = os.path.join(data_path, "report")
reportlets_dir = os.path.join(data_path, "report/reportlets/figures")
all_tables, repetition_times = get_bids_data(data_path)

generate_figure(all_tables, repetition_times, 'global_signal', data_path)
generate_figure(all_tables, repetition_times, 'csf', data_path)

bootstrap_file="./container/bootstrap-fmriprep.yml"
prov = {}

#prov["Freesurfer build stamp"] = f"freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275"
#prov["Execution environment"] = f"Linux d4ef45c6cca2 5.14.0-1057-oem #64-Ubuntu"

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

report_filename = generate_report_with_plots(
    output_dir=output_dir, 
    run_uuid=uuid.uuid4(), 
    reportlets_dir=reportlets_dir,
    bootstrap_file=bootstrap_file,
    metadata=metadata,
    plugin_meta={}
)
