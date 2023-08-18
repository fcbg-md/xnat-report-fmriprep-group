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


def extract_file_info(filename):
    # Extract relevant info from the filename
    parts = filename.split('_')
    subject = [part for part in parts if 'sub' in part][0]
    task = [part.split('-')[1] for part in parts if 'task' in part][0]
    return {'file_name': filename, 'subject': subject, 'task': task}



def generate_figures(all_tables, repetition_times, signal, output_dir):

    file_info_list = []
    fig = go.Figure()

    for table, repetition_time in zip(all_tables, repetition_times):
        df = pd.read_csv(table, sep='\t')
        
        # Vérifier si la colonne 'signal' est dans df
        if signal in df.columns:
            df = df[[signal]]
            base_name = os.path.basename(table)

            file_info = extract_file_info(os.path.basename(table).split('.')[0])

            match = re.match(r"(sub-\w+)_ses-\w+_task-(\w+)_run-\w+_desc-\w+\.tsv", base_name)

            if match:
                subject = match.group(1)
                task = match.group(2)
                file_num = all_tables.index(table) + 1  # supposons que file_num est simplement l'index + 1
                file_info_list.append([file_num, base_name, subject, task])
            
            session_name = file_info.get('ses', 'N/A')

            signal_values = df[signal].values
            time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)
            

            custom_legend = f"{subject}_ses-{session_name}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
            fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
        

    # Convertir la liste en DataFrame
    columns = ["file_num", "file_name", "subject", "task"]
    df_files = pd.DataFrame(file_info_list, columns=columns)

    subjects = df_files['subject'].unique().tolist()
    dfs_by_subject = {subject: df_files[df_files['subject'] == subject] for subject in subjects}

    dfs = {}
    for subject in subjects:
        dfs[subject]=pd.pivot_table(df_files[df_files['subject']==subject],
                                        values=['file_name'],
                                        index=['file_num'],
                                        columns=['task'],
                                        aggfunc=np.sum)
        
    common_cols = []
    common_rows = []
    for df in dfs.keys():
        common_cols = sorted(list(set().union(common_cols,list(dfs[df]))))
        common_rows = sorted(list(set().union(common_rows,list(dfs[df].index))))

    # find dimensionally common dataframe
    df_common = pd.DataFrame(np.nan, index=common_rows, columns=common_cols)

    # reshape each dfs[df] into common dimensions
    dfc={}
    for df_item in dfs:
        #print(dfs[unshaped])
        df1 = dfs[df_item].copy()
        s=df_common.combine_first(df1)
        df_reshaped = df1.reindex_like(s)
        dfc[df_item]=df_reshaped

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df_reshaped)

    # Nom des colonnes
    column_names = df_reshaped.columns.tolist()
    print("Column Names:", column_names)

    # Nombre de colonnes
    num_columns = len(df_reshaped.columns)
    print("Number of Columns:", num_columns)

    # Nombre de lignes
    num_rows = len(df_reshaped)
    print("Number of Rows:", num_rows)



    # one trace for each column per dataframe: AI and RANDOM



# def generate_figures(all_tables, repetition_times, signal, output_dir):

#     fig = go.Figure()

#     # Dictionary to group signal values by subject and session
#     subjects_data = {}

#     for table, repetition_time in zip(all_tables, repetition_times):
#         df = pd.read_csv(table, sep='\t')
#         if signal in df.columns:
#             file_info = extract_file_info(os.path.basename(table).split('.')[0])
            
#             subject_name = file_info['sub']
#             session_name = file_info.get('ses', 'N/A')
            
#             if subject_name not in subjects_data:
#                 subjects_data[subject_name] = {}
#             if session_name not in subjects_data[subject_name]:
#                 subjects_data[subject_name][session_name] = []
            
#             signal_values = df[signal].values
#             time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)
            
#             # Adding to the figure
#             custom_legend = f"{subject_name}_ses-{session_name}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
#             fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
            
#             subjects_data[subject_name][session_name].append((time_indices, signal_values))

#     dropdown_buttons = []
    
#     # "All" option
#     dropdown_buttons.append(dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects and Sessions', 'showlegend': True}]))
    
#     for subject in subjects_data:
#         for session in subjects_data[subject]:
#             label = f"{subject} - Session: {session}"
#             visibility = [trace.name.startswith(f"{subject}_ses-{session}") for trace in fig.data]
#             dropdown_buttons.append(dict(label=label, method='update', args=[{'visible': visibility}, {'title': f'{signal} for {label}', 'showlegend': True}]))

#     fig.update_layout(
#         updatemenus=[
#             dict(buttons=dropdown_buttons, direction="down", x=0.1, xanchor="left", y=1.1, yanchor="top")
#         ]
#     )

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

def generate_figure(all_tables, repetition_times, signal, output_dir):
    tasks = set()

    # Collect unique tasks from all tables using extract_file_info function
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))

    for task in tasks:
        fig = go.Figure()
        subjects_data = {}
        
        for table, repetition_time in zip(all_tables, repetition_times):
            df = pd.read_csv(table, sep='\t')
            
            file_info = extract_file_info(os.path.basename(table).split('.')[0])
            if file_info.get('task') != task:
                continue

            if signal in df.columns:
                file_name = os.path.basename(table).split('.')[0]
                subject_name = file_name.split('_')[0]

                signal_values = df[signal]
                time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)

                if subject_name not in subjects_data:
                    subjects_data[subject_name] = []

                subjects_data[subject_name].append((table, time_indices, signal_values))

        visibility_lists = []

        for subject, data_list in subjects_data.items():
            visibility = [False] * len(all_tables)

            for current_table, time_indices, signal_values in data_list:
                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])
                custom_legend = f"{subject}_ses-{file_info.get('ses', 'N/A')}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
                
                fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
                current_trace_index = len(fig.data) - 1
                visibility[current_trace_index] = True

            visibility_lists.append(visibility)

        # Dropdown menu
        dropdown_buttons = [dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects in task {task}', 'showlegend': True}])]
        
        for i, (subject, _) in enumerate(subjects_data.items()):
            dropdown_buttons.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{signal} for {subject} in task {task}', 'showlegend': True}]))
        
        fig.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig_name = f"desc-{signal}_signal_for_task-{task}.html"
        fig.write_html(os.path.join(output_dir, fig_name))

    return tasks


#     fig_name = f"desc-{signal}_signal_for_all_subjects.html"
#     fig.write_html(os.path.join(output_dir, fig_name))




def generate_figure2(all_tables, repetition_times, signals, output_dir):
    tasks = set()

    # Detect all distinct tasks from tables
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))


    file_colors = ['red', 'green', 'blue', 'orange', 'purple']
    signal_colors = ['black', 'grey', 'brown']

    # For each table and signal, add a trace to the figure
    for i, table in enumerate(all_tables):
        df = pd.read_csv(table, sep='\t')
        for j, signal in enumerate(signals):
            if signal in df.columns:
                signal_values = df[signal]
                repetition_time = repetition_times[i]
                time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 
                file_color = file_colors[i % len(file_colors)]
                signal_color = signal_colors[j % len(signal_colors)]
                color = file_color if len(signals) == 1 else signal_color
                fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal, visible="legendonly"))

    # Dropdown for subjects
    subjects_buttons = []
    for i, subject in enumerate(subject_names):
        visibility = [subject in trace.name for trace in fig.data]
        subjects_buttons.append(dict(label=subject, method="update", args=[{"visible": visibility}]))

    # Dropdown for signals
    signals_buttons = []
    for signal in signals:
        visibility = [signal in trace.name for trace in fig.data]
        signals_buttons.append(dict(label=signal, method="update", args=[{"visible": visibility}]))

    # For each task, generate a distinct figure
    for task in tasks:
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


    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=subjects_buttons,
                direction="down",
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            dict(
                active=0,
                buttons=signals_buttons,
                direction="down",
                x=0.35,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f'Select a Subject and Signal to View',
        xaxis_title='Time (seconds)'
    )

    # Specify the directory to save the file
    output_dir = os.path.join(output_dir)

    # Check if the directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure to an HTML file
    fig_name = "desc-signal_for_all_subjects.html"

    fig_name = f"desc-{signal}_signal_for_task-{task}.html"

    fig.write_html(os.path.join(output_dir, fig_name))
    
    return tasks

# Example use
# generate_figure2(['path_to_file1.csv', 'path_to_file2.csv'], [2.0, 2.0], ['Signal1', 'Signal2'], './output')


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
    tasks,
    **entities
):
    out_filenames = []

    for task in tasks:

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
        
        # Renommer le fichier de sortie pour inclure le nom de la tâche
        original_filename = robj.out_filename.absolute()
        new_filename = os.path.join(output_dir, f"report_{task}.html")
        os.rename(original_filename, new_filename)

        out_filenames.append(new_filename)

    return out_filenames

