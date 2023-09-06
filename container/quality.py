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
import random
from collections import defaultdict


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

from itertools import cycle
import plotly.express as px


def extract_unique_tasks(all_tables):
    tasks = set()
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))
    return tasks


def generate_divergent_colors(n):
    turbo_colors = px.colors.cyclical.HSV
    step = len(turbo_colors) // n
    return [turbo_colors[i] for i in range(0, len(turbo_colors), step)][:n]


def distribute_colors(palette, n):
    step = len(palette) // n
    return [palette[i * step] for i in range(n)]

def generate_figure(all_tables, repetition_times, signal, output_dir):
    tasks = set()
    fig_all = go.Figure() 

    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))


    for task in tasks:
        fig_task = go.Figure()
        subject_data = {}

        
        for table, repetition_time in zip(all_tables, repetition_times):
            df = pd.read_csv(table, sep='\t')
            file_info = extract_file_info(os.path.basename(table).split('.')[0])

            if signal in df.columns:
                subject_name = os.path.basename(table).split('_')[0]
                if subject_name not in subject_data:
                    subject_data[subject_name] = []
                
                signal_values = df[signal]
                time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)

                motion_outliers = [col for col in df.columns if 'motion_outlier' in col]
                subject_data[subject_name].append((table, time_indices, signal_values, motion_outliers))

        visibility_lists = []
        visibility_all_lists = []
        
        for subject, data_list in subject_data.items():
            visibility = [False] * len(all_tables)
            visibility_all = [False] * len(all_tables)

            colors = generate_divergent_colors(len(all_tables))
            

            for i, (current_table, time_indices, signal_values, motion_outliers) in enumerate(data_list):
                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])
                current_subject_name = os.path.basename(current_table).split('_')[0]
                current_session = os.path.basename(current_table).split('_')[1]
                current_task_name = file_info.get('task')
                current_run = os.path.basename(current_table).split('_')[3]


                custom_legend = f"{current_subject_name} - {current_session} - {current_task_name} - {current_run}"

                color = colors[i]


                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])

                palette_cycle = cycle([px.colors.sequential.Plasma, px.colors.sequential.Viridis, px.colors.sequential.Inferno, px.colors.sequential.Magma])
                task_info = {}

                if current_task_name not in task_info:
                    task_info[current_task_name] = {'palette': next(palette_cycle), 'num_traces': 0}
                num_traces = task_info[current_task_name]['num_traces']
                palette = task_info[current_task_name]['palette']
                colors_for_task = distribute_colors(palette, num_traces + 1)

                colortasks = colors_for_task[-1]
                task_info[current_task_name]['num_traces'] += 1

                if file_info.get('task') == task: 
                    fig_task.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend, line=dict(color=color)))

                fig_all.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend, line=dict(color=colortasks)))

                current_trace_index_task = len(fig_task.data) - 1
                current_trace_index_all = len(fig_all.data) - 1

                # Mise à jour de la visibilité pour fig_task
                if current_trace_index_task >= len(visibility):
                    extend_length = (current_trace_index_task - len(visibility) + 1)
                    visibility.extend([False] * extend_length)
                visibility[current_trace_index_task] = True

                # Mise à jour de la visibilité pour fig_all
                if current_trace_index_all >= len(visibility_all):
                    extend_length = (current_trace_index_all - len(visibility_all) + 1)
                    visibility_all.extend([False] * extend_length)
                visibility_all[current_trace_index_all] = True

        
                df_current = pd.read_csv(current_table, sep='\t')
                for motion_outlier in motion_outliers:
                    if motion_outlier in df_current.columns:
                        outlier_indices = np.where(df_current[motion_outlier] == 1)[0]
                        outlier_times = time_indices[outlier_indices]
                        y_values = np.array([np.nan]*len(time_indices))
                        y_values[outlier_indices] = signal_values[outlier_indices]

                        fig_task.add_trace(go.Scatter(x=time_indices, y=y_values, mode='markers', name=f"{custom_legend} - {motion_outlier}", showlegend=False, marker=dict(color=color, size=8)))

                if current_trace_index_task >= len(visibility):
                    extend_length = (current_trace_index_task - len(visibility) + 1)
                    visibility.extend([False] * extend_length)
                visibility[current_trace_index_task] = True

                if current_trace_index_all >= len(visibility_all):
                    extend_length = (current_trace_index_all - len(visibility_all) + 1)
                    visibility_all.extend([False] * extend_length)
                visibility_all[current_trace_index_all] = True

            visibility_lists.append(visibility)
            visibility_all_lists.append(visibility_all)

        yaxis_title = signal

        if signal == 'framewise_displacement':
            yaxis_title = 'FD (mm)'
        elif signal == 'std_dvars':
            yaxis_title = 'Standardized DVARS'


        fig_task.update_layout(
            hoverlabel_namelength=-1,
            title={
                'text': f'{signal} for task {task}',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'{yaxis_title}', 
            autosize=True  
        )

        fig_all.update_layout(
            hoverlabel_namelength=-1,
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


        # Dropdown menu
        dropdown_buttons_tasks = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_task.data)}, {'title': f'{signal} for All Subjects in task {task}', 'showlegend': True}])]
        dropdown_buttons_all = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_all.data)}, {'title': f'{signal} for All Subjects in all tasks', 'showlegend': True}])]

        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_tasks.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{signal} for {subject} in task {task}', 'showlegend': True}]))
        
        fig_task.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_tasks, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        
        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_all_lists[i]}, {'title': f'{signal} for {subject} in all tasks', 'showlegend': True}]))
        
        fig_all.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_all, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig_name = f"desc-tasks{signal}_signal_for_task-{task}.html"
        fig_task.write_html(os.path.join(output_dir, fig_name))
            
        # Sauvegardez fig_all
    fig_all_name = f"desc-all{signal}_signal_for_all_tasks.html"
    fig_all.write_html(os.path.join(output_dir, fig_all_name))


def generate_figures_motion(all_tables, repetition_times, signals, output_dir):
    tasks = set()
    fig_all = go.Figure()

    color_map = ['darkred', 'black', 'grey']

    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))

    for task in tasks:
        fig_task = go.Figure()
        subject_data = {}

        for table, repetition_time in zip(all_tables, repetition_times):
            df = pd.read_csv(table, sep='\t')
            file_info = extract_file_info(os.path.basename(table).split('.')[0])

            subject_name = os.path.basename(table).split('_')[0]
            if subject_name not in subject_data:
                subject_data[subject_name] = []

            time_indices = np.arange(0, len(df) * repetition_time, repetition_time)
            subject_data[subject_name].append((table, time_indices, df))

        visibility_lists = []
        visibility_all_lists = []
            

        for subject, data_list in subject_data.items():
            visibility = [False] * len(all_tables) * len(signals)
            visibility_all = [False] * len(all_tables) * len(signals)

            for current_table, time_indices, df in data_list:
                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])
                current_subject_name = os.path.basename(current_table).split('_')[0]
                current_session = os.path.basename(current_table).split('_')[1]
                current_task_name = file_info.get('task')
                current_run = os.path.basename(current_table).split('_')[3]

                for i, signal in enumerate(signals):
                    if signal in df.columns:
                        custom_legend = f"{current_subject_name} - {current_session} - {current_task_name} - {current_run} - {signal}"
                        color = color_map[i % len(color_map)]

                        if file_info.get('task') == task:
                            fig_task.add_trace(go.Scatter(x=time_indices, y=df[signal], mode='lines', name=custom_legend, line=dict(color=color)))

                        fig_all.add_trace(go.Scatter(x=time_indices, y=df[signal], mode='lines', name=custom_legend, line=dict(color=color)))

                        current_trace_index_task = len(fig_task.data) - 1
                        current_trace_index_all = len(fig_all.data) - 1

                        if current_trace_index_task >= len(visibility):
                            extend_length = current_trace_index_task - len(visibility) + 1
                            visibility.extend([False] * extend_length)
                        visibility[current_trace_index_task] = True

                        if current_trace_index_all >= len(visibility_all):
                            extend_length = current_trace_index_all - len(visibility_all) + 1
                            visibility_all.extend([False] * extend_length)
                        visibility_all[current_trace_index_all] = True

            visibility_lists.append(visibility)
            visibility_all_lists.append(visibility_all)

        if signal == 'rot_z':
            display_sinal = 'Rotation'
        elif signal == 'trans_z':
            display_sinal = 'Translation'

        fig_task.update_layout(
            hoverlabel_namelength=-1,
            title={
                'text': f'{display_sinal} for task {task}',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'{display_sinal}', 
            autosize=True
        )

        fig_all.update_layout(
            hoverlabel_namelength=-1,
            title={
                'text': f'{display_sinal} for all tasks',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'{display_sinal}', 
            autosize=True
        )

        dropdown_buttons_tasks = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_task.data)}, {'title': f'{display_sinal} for all Subjects in task {task}', 'showlegend': True}])]
        dropdown_buttons_all = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_all.data)}, {'title': f'{display_sinal} for all Subjects in all tasks', 'showlegend': True}])]

        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_tasks.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{display_sinal} for {subject} in task {task}', 'showlegend': True}]))
        
        fig_task.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_tasks, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        
        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_all_lists[i]}, {'title': f'{display_sinal} for {subject} in all tasks', 'showlegend': True}]))
        
        fig_all.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_all, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig_name = f"desc-tasks{display_sinal} _for_task-{task}.html"
        fig_task.write_html(os.path.join(output_dir, fig_name))

    fig_all_name = f"desc-all{display_sinal} _for_all_tasks.html"
    fig_all.write_html(os.path.join(output_dir, fig_all_name))




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
    

    original_filename = robj.out_filename.absolute()
    new_filename = os.path.join(output_dir, f"report_fmriprep_group.html")
    os.rename(original_filename, new_filename)

    out_filenames.append(new_filename)

    return out_filenames

