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

def bids_subjects(bids_directory: str):
    layout = BIDSLayout(bids_directory)
    subjects_filenames = layout.get(return_type='filename', target='subject', suffix='T1w', extension='nii.gz')
    subjects_ids = layout.get(return_type='id', target='subject', suffix='T1w')

    pattern = "sub-{sub_id}"

    for subject_filename, sub_id_value in zip(subjects_filenames, subjects_ids):
        in_file = layout.get_file(subject_filename)

        entities = in_file.get_entities()
        entities.pop("extension", None)

        subject_string = pattern.format(sub_id=sub_id_value)
        print(subject_string)  # This will print each subject string, like "sub-s05", "sub-s06", etc.

        report_type = entities.pop("datatype", None)
        report_type = "fs"
        return subject_string


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

def format_name(name):
    parts = name.split('_')
    
    # Finding the run part and reformatting it
    for part in parts:
        if "run-" in part:
            run_number = part.split('-')[-1]  # getting the '001' from 'run-001'
            run_formatted = f"run{run_number}"  # converting 'run-001' to 'run001'
            break
    
    return f"{parts[0]}_{parts[1]}_{run_formatted}"


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



def generate_figure(all_tables, repetition_times, signal, output_dir):
    fs=0.015

    fig = go.Figure()

    # Create a list of subject names
    subject_names = [format_name(os.path.basename(table).split('.')[0]) for table in all_tables]

    # Create a list of visibility lists
    visibility_lists = []

    for i, table in enumerate(all_tables):
        df = pd.read_csv(table, sep='\t')

        if signal in df.columns:
            signal_values = df[signal]

            repetition_time = repetition_times[i]
            time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 

            frequencies, times, Sxx = spectrogram(signal_values, fs, nperseg=3)

            mean_psd = np.mean(10 * np.log10(Sxx), axis=0)

            normalized_mean_psd = (mean_psd - np.min(mean_psd)) / (np.max(mean_psd) - np.min(mean_psd))
            enhanced_mean_psd = normalized_mean_psd ** 2
            mean_enhanced_psd = np.mean(enhanced_mean_psd)
            std_enhanced_psd = np.std(enhanced_mean_psd)

            threshold = mean_enhanced_psd + 2*std_enhanced_psd
            anomaly_indices = np.where(enhanced_mean_psd > threshold)[0]

            grouped_anomaly_indices = group_consecutives(anomaly_indices)

            color_palette = sns.color_palette("hsv", len(grouped_anomaly_indices)).as_hex()

            # Create a new visibility list for this file
            visibility = [False] * len(fig.data)
            
            fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line_color='darkgray', name=subject_names[i]))

            # The last trace added should be visible for this file
            visibility.append(True)

            for j, group in enumerate(grouped_anomaly_indices):
                if group: 
                    start_index = (group[0] * len(signal_values)) // len(times)
                    end_index = (group[-1] * len(signal_values)) // len(times)
                    start_index = max(min(start_index, len(signal_values)-1), 0)
                    end_index = max(min(end_index, len(signal_values)-1), 0)
                    anomalies = signal_values[start_index:end_index+1]
                    time_indices_scaled = time_indices[start_index:end_index+1]

                    if len(time_indices_scaled) == 1 and end_index+1 < len(signal_values):
                        # If there is only one anomaly point, include the next non-anomaly point
                        end_index += 1
                        anomalies = signal_values[start_index:end_index+1]
                        time_indices_scaled = time_indices[start_index:end_index+1]

                    fig.add_trace(go.Scatter(x=time_indices_scaled, y=anomalies, mode='lines', line=dict(color=color_palette[j]), name=f'Anomaly {j+1} in {subject_names[i]}'))

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
        #print(subject_names[i])
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



def generate_figure2(all_tables, repetition_times, signals, output_dir):
    fs=0.015
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

def afficher_aseg(dossier_input, subject):
    """
    Affiche l'image aseg.mgz pour un sujet donné.

    Args:
    - dossier_input (str): Chemin du dossier contenant les données dérivées.
    - subject (str): Identifiant du sujet.

    Returns:
    - None
    """

    # Chemin vers le fichier aseg.mgz
    path_aseg = f"{dossier_input}/derivatives/fmriprep/sourcedata/freesurfer/{subject}/mri/aseg.mgz"

    # Charger l'image
    image_aseg = nib.load(path_aseg).get_fdata()

    # Afficher une coupe axiale médiane
    plt.imshow(image_aseg[image_aseg.shape[0] // 2], cmap="gray")
    plt.title(f"aseg.mgz pour le sujet {subject} - coupe axiale")
    plt.colorbar()
    plt.axis('off')
    plt.show()

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

