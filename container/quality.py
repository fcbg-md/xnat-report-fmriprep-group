import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.signal import spectrogram
from bids import BIDSLayout
import re
import json
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.freesurfer.mghformat import load
from nireports.assembler.report import Report

# def bids_subjects(bids_directory: str):
#     layout = BIDSLayout(bids_directory)
#     subjects_filenames = layout.get(return_type='filename', target='subject', suffix='T1w', extension='nii.gz')
#     subjects_ids = layout.get(return_type='id', target='subject', suffix='T1w')

#     pattern = "sub-{sub_id}"

#     for subject_filename, sub_id_value in zip(subjects_filenames, subjects_ids):
#         in_file = layout.get_file(subject_filename)

#         entities = in_file.get_entities()
#         entities.pop("extension", None)

#         subject_string = pattern.format(sub_id=sub_id_value)
#         print(subject_string)  # This will print each subject string, like "sub-s05", "sub-s06", etc.

#         report_type = entities.pop("datatype", None)
#         report_type = "fs"
#         return subject_string


def get_bids_data(data_path):
    deriv_path = os.path.join(data_path, "derivatives/fmriprep")

    layout = BIDSLayout(data_path, derivatives=True)
    lay=BIDSLayout(data_path)

    all_tables = layout.get(extension='.tsv', suffix='timeseries', scope='derivatives', return_type='filename')
    information_files = lay.get(extension='.json', suffix='bold', return_type='subject')  


    subject_names = []


    entities = {}
    for table in all_tables:
        in_file = layout.get_file(table)
        
        entities = in_file.get_entities()
        entities.pop("extension", None)

    repetition_times = []

    for file_path in information_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        repetition_times.append(data.get('RepetitionTime'))

    return all_tables, entities, repetition_times

# def format_name(name):
#     parts = name.split("_")
    
#     # Initialiser les variables pour stocker les parties du nom
#     subject_part = None
#     session_part = None
#     run_formatted = None

#     # Parcourir chaque partie pour trouver les éléments requis
#     for part in parts:
#         if "sub-" in part:
#             subject_part = part
#         elif "ses-" in part:
#             session_part = part
#         elif "run-" in part:
#             run_number = part.split('-')[-1]  # Obtenir '004' de 'run-004'
#             run_formatted = f"run{run_number}"  # Convertir 'run-004' en 'run004'

#     # Vérifier si toutes les parties nécessaires ont été trouvées
#     if not (subject_part and session_part and run_formatted):
#         raise ValueError(f"Le nom '{name}' ne contient pas toutes les parties nécessaires.")

#     print(f"Nom entré : {name}")
#     print(f"Parties : {parts}")

#     return f"{subject_part}_{session_part}_{run_formatted}"



def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def extract_file_info(file_name):
    parts = file_name.split('_')
    info = {}
    for part in parts:
        if '-' in part:  # Check if the part has the expected format
            key, value = part.split('-')
            info[key] = value
    return info


def generate_figure(all_tables, repetition_times, signal, output_dir):

    fig = go.Figure()

    # Dictionary to group signal values by subject
    subjects_data = {}

    for table, repetition_time in zip(all_tables, repetition_times):
        df = pd.read_csv(table, sep='\t')

        if signal in df.columns:
            file_name = os.path.basename(table).split('.')[0]
            subject_name = file_name.split('_')[0]

            signal_values = df[signal]
            time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)

            if subject_name not in subjects_data:
                subjects_data[subject_name] = []

            subjects_data[subject_name].append((table, time_indices, signal_values))

    print("subject_data" , subjects_data)


    visibility_lists = []

    for subject, data_list in subjects_data.items():
        visibility = [False] * len(all_tables) # Ensure visibility is initialized here for each subject

        for current_table, time_indices, signal_values in data_list:
            file_info = extract_file_info(os.path.basename(current_table).split('.')[0])

            # Create a custom legend using the extracted file info
            custom_legend = f"{subject}_ses-{file_info.get('ses', 'N/A')}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
            
            fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
            current_trace_index = len(fig.data) - 1
            visibility[current_trace_index] = True

        visibility_lists.append(visibility)

    print("visibility_lists" , visibility_lists)    

    # Create the dropdown menu
    dropdown_buttons = [
        dict(
            label="All",
            method='update',
            args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects', 'showlegend': True}]
        )
    ]
    
    for i, (subject, _) in enumerate(subjects_data.items()):
        dropdown_buttons.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{signal} for {subject}', 'showlegend': True}]))
    
    fig.update_layout(updatemenus=[dict(
        active=0, 
        buttons=dropdown_buttons, 
        direction="down", 
        pad={"r": 10, "t": 10}, 
        showactive=True, 
        x=0.1, 
        xanchor="left", 
        y=1.1, 
        yanchor="top"
    )])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_name = f"desc-{signal}_signal_for_all_subjects.html"
    fig.write_html(os.path.join(output_dir, fig_name))





def generate_figure2(all_tables, repetition_times, signals, output_dir):
    fig = go.Figure()

    # Create a list of subject names
    subject_names = [os.path.basename(table).split('.')[0] for table in all_tables]

    # Create a list of visibility lists
    visibility_lists = []

    # Create a list of colors for files and signals
    file_colors = ['red', 'green', 'blue', 'orange', 'purple']
    signal_colors = ['black', 'grey', 'brown']

    for i, table in enumerate(all_tables):
        df = pd.read_csv(table, sep='\t')

        # Create a new visibility list for this file
        visibility = [False] * len(fig.data)
        
        
        for j, signal in enumerate(signals):
            if signal in df.columns:
                signal_values = df[signal]

                repetition_time = repetition_times[i]
                time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 

                # Get a color for this file and this signal
                file_color = file_colors[i % len(file_colors)]
                signal_color = signal_colors[j % len(signal_colors)]
                color = file_color if len(signals) == 1 else signal_color

                fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal))

                # The last trace added should be visible for this file
                visibility.append(True)
        
        # Add the visibility list for this file to the list of visibility lists
        visibility_lists.append(visibility)

    fig.update_layout(title=f'{signal}', 
                      xaxis_title='Time (seconds)', 
                      yaxis_title=f'{signal}', 
                      autosize=True)

    # Create the dropdown menu
    dropdown_buttons = []
    for i, visibility in enumerate(visibility_lists):
        # Extend the visibility list to cover all traces
        visibility += [False] * (len(fig.data) - len(visibility))
        dropdown_buttons.append(dict(label=subject_names[i], method='update', 
                                    args=[{'visible': visibility}, 
                                        {'title': f'{signal} for {subject_names[i]}', 'showlegend': True}]))

    # Add 'All Files' option
    dropdown_buttons.append(dict(label='All group', method='update', 
                                args=[{'visible': [True]*len(fig.data)}, 
                                    {'title': f'{signal} for All Files', 'showlegend': True}]))

    fig.update_layout(updatemenus=[dict(active=len(dropdown_buttons)-1, buttons=dropdown_buttons)])
    fig.update_layout(
    updatemenus=[
        dict(
            active=len(dropdown_buttons)-1, 
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,  # this can be tweaked as per the requirement
            xanchor="left",
            y=1.1,  # placing it a bit above so it's visible
            yanchor="top"
        )
    ],
    )

    # Specify the directory to save the file
    output_dir = os.path.join(output_dir)
    
    # Check if the directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure to an HTML file
    fig_name = f"desc-{signal}_signal_for_all_subjects.html"
    fig.write_html(os.path.join(output_dir, fig_name))


def display_motion_outliers(all_files):
    # Loop over all files in the list
    for filepath in all_files:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(filepath, sep='\t')

        # Filter the DataFrame for columns that start with 'motion_outlier'
        motion_outliers = [col for col in df.columns if 'motion_outlier' in col]

        # Display each motion outlier column
        for outlier in motion_outliers:
            print(f"{outlier}:\n")
            print(df[outlier])
            print("\n")


def generate_report_with_plots(
    output_dir,
    run_uuid,
    reportlets_dir,
    bootstrap_file,
    metadata,
    plugin_meta,
    **entities
):
    robj = Report(
        output_dir,
        run_uuid,
        reportlets_dir=reportlets_dir,
        bootstrap_file=bootstrap_file,
        metadata=metadata,
        plugin_meta=plugin_meta,
        **entities,
    )
    robj.generate_report()
    return robj.out_filename.absolute()

