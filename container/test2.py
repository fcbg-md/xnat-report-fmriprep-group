        #     import os
        #     import pandas as pd
        #     import plotly.graph_objects as go
        #     import numpy as np
        #     import seaborn as sns
        #     from plotly.subplots import make_subplots
        #     from scipy.signal import spectrogram
        #     from bids import BIDSLayout
        #     import re
        #     import json
        #     import nibabel as nib
        #     import matplotlib.pyplot as plt
        #     from nibabel.freesurfer.mghformat import load
        #     from nireports.assembler.report import Report

        #     # def bids_subjects(bids_directory: str):
        #     # layout = BIDSLayout(bids_directory)
        #     # subjects_filenames = layout.get(return_type='filename', target='subject', suffix='T1w', extension='nii.gz')
        #     # subjects_ids = layout.get(return_type='id', target='subject', suffix='T1w')

        #     # pattern = "sub-{sub_id}"

        #     # for subject_filename, sub_id_value in zip(subjects_filenames, subjects_ids):
        #     # in_file = layout.get_file(subject_filename)

        #     # entities = in_file.get_entities()
        #     # entities.pop("extension", None)

        #     # subject_string = pattern.format(sub_id=sub_id_value)
        #     # print(subject_string) # This will print each subject string, like "sub-s05", "sub-s06", etc.

        #     # report_type = entities.pop("datatype", None)
        #     # report_type = "fs"
        #     # return subject_string

        #     def get_bids_data(data_path):
        #     deriv_path = os.path.join(data_path, "derivatives/fmriprep")

        #     layout = BIDSLayout(data_path, derivatives=True)
        #     lay=BIDSLayout(data_path)

        #     all_tables = layout.get(extension='.tsv', suffix='timeseries', scope='derivatives', return_type='filename')
        #     information_files = lay.get(extension='.json', suffix='bold', return_type='subject') 

        #     subject_names = []

        #     entities = {}
        #     for table in all_tables:
        #     in_file = layout.get_file(table)
        #     entities = in_file.get_entities()
        #     entities.pop("extension", None)

        #     repetition_times = []

        #     for file_path in information_files:
        #     with open(file_path, 'r') as f:
        #     data = json.load(f)
        #     repetition_times.append(data.get('RepetitionTime'))

        #     return all_tables, entities, repetition_times

        #     # def format_name(name):
        #     # parts = name.split("_")
        #     # # Initialiser les variables pour stocker les parties du nom
        #     # subject_part = None
        #     # session_part = None
        #     # run_formatted = None

        #     # # Parcourir chaque partie pour trouver les éléments requis
        #     # for part in parts:
        #     # if "sub-" in part:
        #     # subject_part = part
        #     # elif "ses-" in part:
        #     # session_part = part
        #     # elif "run-" in part:
        #     # run_number = part.split('-')[-1] # Obtenir '004' de 'run-004'
        #     # run_formatted = f"run{run_number}" # Convertir 'run-004' en 'run004'

        #     # # Vérifier si toutes les parties nécessaires ont été trouvées
        #     # if not (subject_part and session_part and run_formatted):
        #     # raise ValueError(f"Le nom '{name}' ne contient pas toutes les parties nécessaires.")

        #     # print(f"Nom entré : {name}")
        #     # print(f"Parties : {parts}")

        #     # return f"{subject_part}_{session_part}_{run_formatted}"


        #     def group_consecutives(vals, step=1):
        #     """Return list of consecutive lists of numbers from vals (number list)."""
        #     run = []
        #     result = [run]
        #     expect = None
        #     for v in vals:
        #     if (v == expect) or (expect is None):
        #     run.append(v)
        #     else:
        #     run = [v]
        #     result.append(run)
        #     expect = v + step
        #     return result

        #     def extract_file_info(filename):
        #     # Extract relevant info from the filename
        #     parts = filename.split('_')
        #     subject = [part for part in parts if 'sub' in part][0]
        #     task = [part.split('-')[1] for part in parts if 'task' in part][0]
        #     return {'file_name': filename, 'subject': subject, 'task': task}


        #     # one trace for each column per dataframe: AI and RANDOM


        #     # def generate_figures(all_tables, repetition_times, signal, output_dir):

        #     # fig = go.Figure()

        #     # # Dictionary to group signal values by subject and session
        #     # subjects_data = {}

        #     # for table, repetition_time in zip(all_tables, repetition_times):
        #     # df = pd.read_csv(table, sep='\t')
        #     # if signal in df.columns:
        #     # file_info = extract_file_info(os.path.basename(table).split('.')[0])
        #     # subject_name = file_info['sub']
        #     # session_name = file_info.get('ses', 'N/A')
        #     # if subject_name not in subjects_data:
        #     # subjects_data[subject_name] = {}
        #     # if session_name not in subjects_data[subject_name]:
        #     # subjects_data[subject_name][session_name] = []
        #     # signal_values = df[signal].values
        #     # time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)
        #     # # Adding to the figure
        #     # custom_legend = f"{subject_name}_ses-{session_name}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
        #     # fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
        #     # subjects_data[subject_name][session_name].append((time_indices, signal_values))

        #     # dropdown_buttons = []
        #     # # "All" option
        #     # dropdown_buttons.append(dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects and Sessions', 'showlegend': True}]))
        #     # for subject in subjects_data:
        #     # for session in subjects_data[subject]:
        #     # label = f"{subject} - Session: {session}"
        #     # visibility = [trace.name.startswith(f"{subject}_ses-{session}") for trace in fig.data]
        #     # dropdown_buttons.append(dict(label=label, method='update', args=[{'visible': visibility}, {'title': f'{signal} for {label}', 'showlegend': True}]))

        #     # fig.update_layout(
        #     # updatemenus=[
        #     # dict(buttons=dropdown_buttons, direction="down", x=0.1, xanchor="left", y=1.1, yanchor="top")
        #     # ]
        #     # )

        #     # if not os.path.exists(output_dir):
        #     # os.makedirs(output_dir)

        #     def generate_figure(all_tables, repetition_times, signal, output_dir):
        #     tasks = set()

        #     # Collect unique tasks from all tables using extract_file_info function
        #     for table in all_tables:
        #     file_info = extract_file_info(os.path.basename(table).split('.')[0])
        #     tasks.add(file_info.get('task', 'N/A'))

        #     for task in tasks:
        #     fig = go.Figure()
        #     subjects_data = {}
        #     for table, repetition_time in zip(all_tables, repetition_times):
        #     df = pd.read_csv(table, sep='\t')
        #     file_info = extract_file_info(os.path.basename(table).split('.')[0])
        #     if file_info.get('task') != task:
        #     continue

        #     if signal in df.columns:
        #     file_name = os.path.basename(table).split('.')[0]
        #     subject_name = file_name.split('_')[0]

        #     signal_values = df[signal]
        #     time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)

        #     if subject_name not in subjects_data:
        #     subjects_data[subject_name] = []

        #     subjects_data[subject_name].append((table, time_indices, signal_values))

        #     visibility_lists = []

        #     fig.update_layout(
        #     title={
        #     'text': f'{signal} for task: {task}',
        #     'y':0.95,
        #     'x':0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'},
        #     title_font=dict(size=24, color='rgb(107, 107, 107)', family="Courier New, monospace"),
        #     xaxis_title='Time (seconds)', 
        #     yaxis_title=f'{signal}', 
        #     autosize=True
        #     )

        #     for subject, data_list in subjects_data.items():
        #     visibility = [False] * len(all_tables)

        #     for current_table, time_indices, signal_values in data_list:
        #     file_info = extract_file_info(os.path.basename(current_table).split('.')[0])
        #     custom_legend = f"{subject}_ses-{file_info.get('ses', 'N/A')}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
        #     fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
        #     current_trace_index = len(fig.data) - 1
        #     visibility[current_trace_index] = True

        #     visibility_lists.append(visibility)

        #     # Dropdown menu
        #     dropdown_buttons = [dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects in task {task}', 'showlegend': True}])]
        #     for i, (subject, _) in enumerate(subjects_data.items()):
        #     dropdown_buttons.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{signal} for {subject} in task {task}', 'showlegend': True}]))
        #     fig.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        #     if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        #     fig_name = f"desc-{signal}_signal_for_task-{task}.html"
        #     fig.write_html(os.path.join(output_dir, fig_name))

        #     return tasks


        #     # def generate_figure2(all_tables, repetition_times, signals, output_dir):
        #     # tasks = set()
        #     # num_tasks = len(all_tables)
        #     # fig = make_subplots(rows=num_tasks, cols=1, shared_xaxes=True, vertical_spacing=0.007)

        #     # # Pour stocker toutes les options de boutons de menu déroulant pour chaque tâche
        #     # all_dropdown_buttons = []

        #     # # Pour chaque table (chaque tâche), générez un subplot
        #     # for task_idx, table in enumerate(all_tables):
        #     # file_info = extract_file_info(os.path.basename(table).split('.')[0])
        #     # task_name = file_info.get('task', 'N/A')
        #     # tasks.add(task_name)

        #     # # Créer une liste de noms de sujets
        #     # subject_names = [os.path.basename(table).split('.')[0] for table in all_tables]

        #     # # Créer une liste de listes de visibilité
        #     # visibility_lists = []

        #     # # Create the dropdown menu (MOVED THIS INSIDE THE LOOP)
        #     # dropdown_buttons = []

        #     # for i, subject_table in enumerate(all_tables):
        #     # df = pd.read_csv(subject_table, sep='\t')
        #     # # Créer une nouvelle liste de visibilité pour ce fichier
        #     # visibility = [False] * len(fig.data)

        #     # for j, signal in enumerate(signals):
        #     # if signal in df.columns:
        #     # signal_values = df[signal]
        #     # repetition_time = repetition_times[task_idx]
        #     # time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 

        #     # # Ajouter une trace pour ce signal
        #     # fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=f"{subject_names[i]} {signal}"), row=task_idx+1, col=1)

        #     # # La dernière trace ajoutée devrait être visible pour ce fichier
        #     # visibility.append(True)

        #     # # Ajouter le bouton pour ce sujet
        #     # dropdown_buttons.append(dict(label=subject_names[i], method="update", args=[{"visible": visibility}, {}]))

        #     # # Ajouter la liste de visibilité pour ce sujet à la liste des listes de visibilité
        #     # visibility_lists.append(visibility)
        #     # all_dropdown_buttons.append(dropdown_buttons)

        #     # fig.update_layout(title=f'{signal}', 
        #     # xaxis_title='Time (seconds)', 
        #     # yaxis_title=f'{signal}', 
        #     # autosize=True)

        #     # # Create the dropdown menu
        #     # dropdown_buttons = []
        #     # for i, visibility in enumerate(visibility_lists):
        #     # # Extend the visibility list to cover all traces
        #     # visibility += [False] * (len(fig.data) - len(visibility))
        #     # dropdown_buttons.append(dict(label=subject_names[i], method='update', 
        #     # args=[{'visible': visibility}, 
        #     # {'title': f'{signal} for {subject_names[i]}', 'showlegend': True}]))

        #     # # Appliquer les mises à jour de mise en page et les boutons de menu déroulant à la figure
        #     # for task_idx, task_name in enumerate(tasks):
        #     # fig.update_layout(
        #     # title={
        #     # 'text': f'Signals for Task: {task_name}',
        #     # 'y': 0.95,
        #     # 'x': 0.5,
        #     # 'xanchor': 'center',
        #     # 'yanchor': 'top'
        #     # },
        #     # title_font=dict(size=24, color='rgb(107, 107, 107)', family="Courier New, monospace"),
        #     # xaxis_title='Time (seconds)',
        #     # yaxis_title=f'Signal Value',
        #     # autosize=True
        #     # )

        #     # fig.update_layout(updatemenus=[dict(buttons=all_dropdown_buttons[task_idx])])

        #     # # Spécifier le répertoire pour sauvegarder le fichier
        #     # if not os.path.exists(output_dir):
        #     # os.makedirs(output_dir)

        #     # fig_name = f"desc-signal_for_{'_'.join(tasks)}.html"
        #     # fig.write_html(os.path.join(output_dir, fig_name))

        #     # return tasks

        #     def generate_figure2(all_tables, repetition_times, signals, output_dir):
        #     tasks = set()

        #     # Detect all distinct tasks from tables
        #     for table in all_tables:
        #     file_info = extract_file_info(os.path.basename(table).split('.')[0])
        #     tasks.add(file_info.get('task', 'N/A'))

        #     # For each task, generate a distinct figure
        #     for task in tasks:
        #     fig = go.Figure()

        #     # Create a list of subject names
        #     subject_names = [os.path.basename(table).split('.')[0] for table in all_tables]

        #     # Create a list of visibility lists
        #     visibility_lists = []

        #     # Create a list of colors for files and signals
        #     file_colors = ['red', 'green', 'blue', 'orange', 'purple']
        #     signal_colors = ['black', 'grey', 'brown']

        #     for i, table in enumerate(all_tables):
        #     df = pd.read_csv(table, sep='\t')

        #     # Create a new visibility list for this file
        #     visibility = [False] * len(fig.data)
        #     for j, signal in enumerate(signals):
        #     if signal in df.columns:
        #     signal_values = df[signal]

        #     repetition_time = repetition_times[i]
        #     time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 

        #     # Get a color for this file and this signal
        #     file_color = file_colors[i % len(file_colors)]
        #     signal_color = signal_colors[j % len(signal_colors)]
        #     color = file_color if len(signals) == 1 else signal_color

        #     fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal))

        #     # The last trace added should be visible for this file
        #     visibility.append(True)
        #     # Add the visibility list for this file to the list of visibility lists
        #     visibility_lists.append(visibility)
        #     fig.update_layout(
        #     title={
        #     'text': f'{signal} for task: {task}',
        #     'y':0.95,
        #     'x':0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'},
        #     title_font=dict(size=24, color='rgb(107, 107, 107)', family="Courier New, monospace"),
        #     xaxis_title='Time (seconds)', 
        #     yaxis_title=f'{signal}', 
        #     autosize=True
        #     )

        #     # Create the dropdown menu
        #     dropdown_buttons = []
        #     for i, visibility in enumerate(visibility_lists):
        #     # Extend the visibility list to cover all traces
        #     visibility += [False] * (len(fig.data) - len(visibility))
        #     dropdown_buttons.append(dict(label=subject_names[i], method='update', 
        #     args=[{'visible': visibility}, 
        #     {'title': f'{signal} for {subject_names[i]}', 'showlegend': True}]))

        #     # Add 'All Files' option
        #     dropdown_buttons.append(dict(label='All group', method='update', 
        #     args=[{'visible': [True]*len(fig.data)}, 
        #     {'title': f'{signal} for All Files', 'showlegend': True}]))

        #     fig.update_layout(updatemenus=[dict(active=len(dropdown_buttons)-1, buttons=dropdown_buttons)])
        #     fig.update_layout(
        #     updatemenus=[
        #     dict(
        #     active=len(dropdown_buttons)-1, 
        #     buttons=dropdown_buttons,
        #     direction="down",
        #     pad={"r": 10, "t": 10},
        #     showactive=True,
        #     x=0.1, # this can be tweaked as per the requirement
        #     xanchor="left",
        #     y=1.1, # placing it a bit above so it's visible
        #     yanchor="top"
        #     )
        #     ],
        #     )

        #     # Specify the directory to save the file
        #     output_dir = os.path.join(output_dir)
        #     # Check if the directory exists, if not create it
        #     if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        #     # Save the figure to an HTML file
        #     fig_name = f"desc-{signal}_signal_for_all_subjects.html"
        #     fig.write_html(os.path.join(output_dir, fig_name))

        #     return tasks

        #     # Example use
        #     # generate_figure2(['path_to_file1.csv', 'path_to_file2.csv'], [2.0, 2.0], ['Signal1', 'Signal2'], './output')

        #     # def display_motion_outliers(all_files):
        #     # # Loop over all files in the list
        #     # for filepath in all_files:
        #     # # Read the file into a pandas DataFrame
        #     # df = pd.read_csv(filepath, sep='\t')

        #     # # Filter the DataFrame for columns that start with 'motion_outlier'
        #     # motion_outliers = [col for col in df.columns if 'motion_outlier' in col]

        #     # # Display each motion outlier column
        #     # for outlier in motion_outliers:
        #     # print(f"{outlier}:\n")
        #     # print(df[outlier])
        #     # print("\n")

        #     def generate_report_with_plots(
        #     output_dir,
        #     run_uuid,
        #     reportlets_dir,
        #     bootstrap_file,
        #     metadata,
        #     plugin_meta,
        #     tasks,
        #     **entities
        #     ):
        #     out_filenames = []

        #     for task in tasks:

        #     robj = Report(
        #     output_dir,
        #     run_uuid,
        #     reportlets_dir=reportlets_dir,
        #     bootstrap_file=bootstrap_file,
        #     metadata=metadata,
        #     plugin_meta=plugin_meta,
        #     **entities,
        #     )
        #     robj.generate_report()
        #     # Renommer le fichier de sortie pour inclure le nom de la tâche
        #     original_filename = robj.out_filename.absolute()
        #     new_filename = os.path.join(output_dir, f"report_fmriprep_group.html")
        #     os.rename(original_filename, new_filename)

        #     out_filenames.append(new_filename)

        # return out_filenames

