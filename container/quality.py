import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
#from scipy.signal import spectrogram
from bids import BIDSLayout
import json
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


def extract_file_info(filename):
    # Extract relevant info from the filename
    parts = filename.split('_')
    subject = [part for part in parts if 'sub' in part][0]
    task = [part.split('-')[1] for part in parts if 'task' in part][0]
    return {'file_name': filename, 'subject': subject, 'task': task}



    # one trace for each column per dataframe: AI and RANDOM


def detect_outliers(all_tables, repetition_times):
    all_outliers_fd = []
    all_outliers_dvars = []
    
    for table, repetition_time in zip(all_tables, repetition_times):
        df = pd.read_csv(table, sep='\t')
        
        SEUIL_FRAMEWISE_DISPLACEMENT = 0.5
        SEUIL_STD_DVARS = 1.5
        
        # Identification des outliers pour "framewise_displacement"
        outliers_fd = df[df['framewise_displacement'] > SEUIL_FRAMEWISE_DISPLACEMENT]
        
        # Identification des outliers pour "std_dvars"
        outliers_dvars = df[df['std_dvars'] > SEUIL_STD_DVARS]
        
        # Stockage des outliers pour chaque tableau
        all_outliers_fd.append(outliers_fd)
        all_outliers_dvars.append(outliers_dvars)
    
    return all_outliers_fd, all_outliers_dvars

def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def extract_unique_tasks(all_tables):
    tasks = set()
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))
    return tasks

def read_and_preprocess_data(task, all_tables, repetition_times, signal):
    global_data = {}  # Dictionnaire global pour contenir toutes les informations
    
    for table, repetition_time in zip(all_tables, repetition_times):
        # Extraire les informations du fichier
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        
        # Filtrer par tâche
        if file_info.get('task') != task:
            continue
        
        # Lire le tableau
        df = pd.read_csv(table, sep='\t')
        
        # Vérifier si le signal est dans le tableau
        if signal in df.columns:
            subject_name = os.path.basename(table).split('_')[0]
            task_name = file_info['task']
            session = os.path.basename(table).split('_')[1]
            run = os.path.basename(table).split('_')[3]
            
            # Initialiser le sujet s'il n'existe pas déjà
            if subject_name not in global_data:
                global_data[subject_name] = {'sessions': {}}
            
            # Initialiser la session s'il n'existe pas déjà
            if session not in global_data[subject_name]['sessions']:
                global_data[subject_name]['sessions'][session] = {'tasks': {}}
                
            # Initialiser la tâche s'il n'existe pas déjà
            if task_name not in global_data[subject_name]['sessions'][session]['tasks']:
                global_data[subject_name]['sessions'][session]['tasks'][task_name] = {'runs': {}}
            
            # Initialiser le run s'il n'existe pas déjà
            if run not in global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs']:
                global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs'][run] = []
            
            # Obtenir les données du signal
            signal_values = df[signal]
            time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)
            
            # Ajouter ces données au dictionnaire global
            global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs'][run].append((table, time_indices, signal_values))
            
    return global_data



def plot_trace_data(fig, fig_tasks, global_data):
    for subject, subject_info in global_data.items():
        for session, session_info in subject_info['sessions'].items():
            for task, task_info in session_info['tasks'].items():
                for run, data_list in task_info['runs'].items():
                    for current_table, time_indices, signal_values in data_list:
                        # Utiliser les valeurs exactes pour la légende
                        custom_legend = f"{subject}_{session}_task-{task}_{run}"
                        fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
                        fig_tasks.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))

def configure_layout_and_interactivity(fig, fig_tasks, task, signal, visibility_lists, visibility_lists_tasks, global_data):
    
    fig_tasks.update_layout(
        title={
            'text': f'{signal} for {task}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
        xaxis_title='Time (seconds)',
        yaxis_title=f'{signal}',
        autosize=True
    )

    fig_tasks.update_layout(
        sliders=[
            {
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Threshold:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [{
                    'args': [
                        {'frame': {'duration': 300, 'redraw': True}},
                        {'mode': 'immediate', 'transition': {'duration': 300}}
                    ],
                    'label': f"{round(threshold, 1)}",
                    'method': 'animate',
                    'value': f"{round(threshold, 1)}"
                } for threshold in np.arange(0, 2, 0.1)]
            }
        ],
    )


    fig.update_layout(
        title={
            'text': f'{signal} for all tasks',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
        xaxis_title='Time (seconds)', 
        yaxis_title=f'{signal}', 
        autosize=True
    )

    fig.update_layout(
    sliders=[
        {
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Threshold:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [
                    {'frame': {'duration': 300, 'redraw': True}},
                    {'mode': 'immediate', 'transition': {'duration': 300}}
                ],
                'label': f"{round(threshold, 1)}",
                'method': 'animate',
                'value': f"{round(threshold, 1)}"
            } for threshold in np.arange(0, 2, 0.1)]
        }
    ],
)


    
    # Dropdown menu
    dropdown_buttons = [dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects in task {task}', 'showlegend': True}])]
    for i, (subject, _) in enumerate(global_data.items()):
        dropdown_buttons.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{signal} for {subject} in task {task}', 'showlegend': True}]))
    fig.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
    fig_tasks.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

def generate_figure(all_tables, repetition_times, signal, output_dir):
    if not all_tables:
        raise ValueError("all_tables must contain at least one element.")
    
    create_output_directory(output_dir)
    tasks = extract_unique_tasks(all_tables)

    for task in tasks:
        fig = go.Figure()
        fig_tasks = go.Figure()
        global_data = read_and_preprocess_data(task, all_tables, repetition_times, signal)
        
        # Appeler plot_trace_data pour fig et fig_tasks
        plot_trace_data(fig, fig_tasks, global_data)

        visibility_lists = []  # Initialize visibility_lists for fig
        visibility_lists_tasks = []  # Initialize visibility_lists for fig_tasks

        for subject, subject_info in global_data.items():
            visibility = [False] * len(fig.data)
            visibility_tasks = [False] * len(fig_tasks.data)  # For fig_tasks

            for session, session_info in subject_info['sessions'].items():
                for run, data_list in session_info.items():
                    current_trace_index = len(fig.data) - 1
                    visibility[current_trace_index] = True

                    # Do the same for fig_tasks
                    current_trace_index_tasks = len(fig_tasks.data) - 1
                    visibility_tasks[current_trace_index_tasks] = True

            visibility_lists.append(visibility)
            visibility_lists_tasks.append(visibility_tasks)  # For fig_tasks

        configure_layout_and_interactivity(fig, fig_tasks, task, signal, visibility_lists, visibility_lists_tasks, global_data)
        
        fig_name = f"desc-{signal}_signal_for_task-{task}.html"
        fig.write_html(os.path.join(output_dir, fig_name))
        
        # If you want to save fig_tasks
        fig_tasks_name = f"desc-{signal}_signal_tasks_for_task-{task}.html"
        fig_tasks.write_html(os.path.join(output_dir, fig_tasks_name))

    return tasks



def generate_figure2(all_tables, repetition_times, signals, output_dir):
    tasks = set()
    global_data = {}

    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))

    # For each task, generate a distinct figure
    for task in tasks:
        fig = go.Figure()
        fig_tasks = go.Figure()

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

                                        # Obtenir les données du signal
                    signal_values = df[signal]

                    repetition_time = repetition_times[i]
                    time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time)
                    subject_name = os.path.basename(table).split('_')[0]
                    task_name = file_info['task']
                    session = os.path.basename(table).split('_')[1]
                    run = os.path.basename(table).split('_')[3]
                    
                    # Initialiser le sujet s'il n'existe pas déjà
                    if subject_name not in global_data:
                        global_data[subject_name] = {'sessions': {}}
                    
                    # Initialiser la session s'il n'existe pas déjà
                    if session not in global_data[subject_name]['sessions']:
                        global_data[subject_name]['sessions'][session] = {'tasks': {}}
                        
                    # Initialiser la tâche s'il n'existe pas déjà
                    if task_name not in global_data[subject_name]['sessions'][session]['tasks']:
                        global_data[subject_name]['sessions'][session]['tasks'][task_name] = {'runs': {}}
                    
                    # Initialiser le run s'il n'existe pas déjà
                    if run not in global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs']:
                        global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs'][run] = []
                    
                    
                    # Ajouter ces données au dictionnaire global
                    global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs'][run].append((table, time_indices, signal_values))


                    # Get a color for this file and this signal
                    file_color = file_colors[i % len(file_colors)]
                    signal_color = signal_colors[j % len(signal_colors)]
                    color = file_color if len(signals) == 1 else signal_color

                    fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal))
                    fig_tasks.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal))

                    # The last trace added should be visible for this file
                    visibility.append(True)
            
            # Add the visibility list for this file to the list of visibility lists
            visibility_lists.append(visibility)

            if signal == "rot_z":
                display_signal = "rotation"
            elif signal == "trans_z":
                display_signal = "translation"
        
        fig_tasks.update_layout(
            title={
                'text': f'{display_signal} for all tasks',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'{display_signal}', 
            autosize=True
        )

        fig.update_layout(
            title={
                'text': f'{display_signal} for all tasks',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'{display_signal}', 
            autosize=True
        )


        # Create the dropdown menu
        dropdown_buttons = []
        
        dropdown_buttons = [dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{display_signal} for All Subjects in task {task}', 'showlegend': True}])]
        for i, (subject, _) in enumerate(global_data.items()):
            dropdown_buttons.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{display_signal} for {subject} in task {task}', 'showlegend': True}]))
        fig.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        fig_tasks.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        
        # # Add 'All Files' option
        # dropdown_buttons.append(dict(label='All group', method='update', 
        #                             args=[{'visible': [True]*len(fig.data)}, 
        #                                 {'title': f'{display_signal} for All Files', 'showlegend': True}]))
        # for i, visibility in enumerate(visibility_lists):
        #     # Extend the visibility list to cover all traces
        #     visibility += [False] * (len(fig.data) - len(visibility))
        #     dropdown_buttons.append(dict(label=subject_names[i], method='update', 
        #                                 args=[{'visible': visibility}, 
        #                                     {'title': f'{display_signal} for {subject_names[i]}', 'showlegend': True}]))



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

        fig_tasks.update_layout(updatemenus=[dict(active=len(dropdown_buttons)-1, buttons=dropdown_buttons)])
        fig_tasks.update_layout(
            width=800,
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

        fig_tasks_name = f"desc-{signal}_signal_for_task-{task}.html"
        fig_tasks.write_html(os.path.join(output_dir, fig_tasks_name))

        fig_name = f"desc-{signal}_signal_for_all_tasks.html"
        fig.write_html(os.path.join(output_dir, fig_name))
    
    return tasks


# def generate_figure2(all_tables, repetition_times, signals, output_dir):
#     tasks = set()

#     # Detect all distinct tasks from tables
#     for table in all_tables:
#         file_info = extract_file_info(os.path.basename(table).split('.')[0])
#         tasks.add(file_info.get('task', 'N/A'))


#     file_colors = ['red', 'green', 'blue', 'orange', 'purple']
#     signal_colors = ['black', 'grey', 'brown']

#     # For each table and signal, add a trace to the figure
#     for i, table in enumerate(all_tables):
#         df = pd.read_csv(table, sep='\t')
#         for j, signal in enumerate(signals):
#             if signal in df.columns:
#                 signal_values = df[signal]
#                 repetition_time = repetition_times[i]
#                 time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 
#                 file_color = file_colors[i % len(file_colors)]
#                 signal_color = signal_colors[j % len(signal_colors)]
#                 color = file_color if len(signals) == 1 else signal_color
#                 fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal, visible="legendonly"))

#     # Dropdown for subjects
#     subjects_buttons = []
#     for i, subject in enumerate(subject_names):
#         visibility = [subject in trace.name for trace in fig.data]
#         subjects_buttons.append(dict(label=subject, method="update", args=[{"visible": visibility}]))

#     # Dropdown for signals
#     signals_buttons = []
#     for signal in signals:
#         visibility = [signal in trace.name for trace in fig.data]
#         signals_buttons.append(dict(label=signal, method="update", args=[{"visible": visibility}]))

#     # For each task, generate a distinct figure
#     for task in tasks:
#         fig = go.Figure()

#         # Create a list of subject names
#         subject_names = [os.path.basename(table).split('.')[0] for table in all_tables]

#         # Create a list of visibility lists
#         visibility_lists = []

#         # Create a list of colors for files and signals
#         file_colors = ['red', 'green', 'blue', 'orange', 'purple']
#         signal_colors = ['black', 'grey', 'brown']

#         for i, table in enumerate(all_tables):
#             df = pd.read_csv(table, sep='\t')

#             # Create a new visibility list for this file
#             visibility = [False] * len(fig.data)
            
            
#             for j, signal in enumerate(signals):
#                 if signal in df.columns:
#                     signal_values = df[signal]

#                     repetition_time = repetition_times[i]
#                     time_indices = np.arange(0, len(signal_values)*repetition_time, repetition_time) 

#                     # Get a color for this file and this signal
#                     file_color = file_colors[i % len(file_colors)]
#                     signal_color = signal_colors[j % len(signal_colors)]
#                     color = file_color if len(signals) == 1 else signal_color

#                     fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', line=dict(color=color), name=subject_names[i]+' '+signal))

#                     # The last trace added should be visible for this file
#                     visibility.append(True)
            
#             # Add the visibility list for this file to the list of visibility lists
#             visibility_lists.append(visibility)


#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 active=0,
#                 buttons=subjects_buttons,
#                 direction="down",
#                 x=0.1,
#                 xanchor="left",
#                 y=1.15,
#                 yanchor="top"
#             ),
#             dict(
#                 active=0,
#                 buttons=signals_buttons,
#                 direction="down",
#                 x=0.35,
#                 xanchor="left",
#                 y=1.15,
#                 yanchor="top"
#             )
#         ],
#         title=f'Select a Subject and Signal to View',
#         xaxis_title='Time (seconds)'
#     )

#     # Specify the directory to save the file
#     output_dir = os.path.join(output_dir)

#     # Check if the directory exists, if not create it
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Save the figure to an HTML file
#     fig_name = "desc-signal_for_all_subjects.html"

#     fig_name = f"desc-{signal}_signal_for_task-{task}.html"

#     fig.write_html(os.path.join(output_dir, fig_name))
    
#     return tasks

# Example use
# generate_figure2(['path_to_file1.csv', 'path_to_file2.csv'], [2.0, 2.0], ['Signal1', 'Signal2'], './output')


# def display_motion_outliers(all_files):
#     # Loop over all files in the list
#     for filepath in all_files:
#         # Read the file into a pandas DataFrame
#         df = pd.read_csv(filepath, sep='\t')

#         # Filter the DataFrame for columns that start with 'motion_outlier'
#         motion_outliers = [col for col in df.columns if 'motion_outlier' in col]

#         # Display each motion outlier column
#         for outlier in motion_outliers:
#             print(f"{outlier}:\n")
#             print(df[outlier])
#             print("\n")


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
        new_filename = os.path.join(output_dir, f"report_fmriprep_group.html")
        os.rename(original_filename, new_filename)

        out_filenames.append(new_filename)

    return out_filenames

