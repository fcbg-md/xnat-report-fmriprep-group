import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from bids import BIDSLayout
import json
from nireports.assembler.report import Report

def get_bids_data(data_path):
    """
    Fetches and returns BIDS (Brain Imaging Data Structure) data from a given path.
    
    Parameters:
    - data_path (str): The file path to the root directory containing BIDS-formatted data.
    
    Returns:
    - all_tables (list): A list of filenames for .tsv files containing time series data.
    - entities (dict): A dictionary containing BIDS entities (e.g., subject, session, run) for the last processed table.
    - repetition_times (list): A list of repetition times extracted from .json files associated with fMRI data.
    """
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
    """
    Extracts relevant information from a BIDS-formatted filename.
    
    Parameters:
    - filename (str): The BIDS-formatted filename to extract information from.
    
    Returns:
    - dict: A dictionary containing extracted information including file name, subject ID, and task name.
    """
    parts = filename.split('_')
    subject = [part for part in parts if 'sub' in part][0]
    task = [part.split('-')[1] for part in parts if 'task' in part][0]
    return {'file_name': filename, 'subject': subject, 'task': task}

def create_output_directory(output_dir):
    """
    Creates an output directory if it doesn't already exist.
    
    Parameters:
    - output_dir (str): The path to the directory that should be created.
    
    Returns:
    - None: The function has the side-effect of creating a directory but does not return any value.
    
    Note:
    - The function uses Python's os.makedirs to create the directory along with any necessary parent directories.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def extract_unique_tasks(all_tables):
    """
    Extracts and returns the unique task names from a list of table filenames.

    Parameters:
    - all_tables (list of str): A list containing the file paths for the timeseries tables.

    Returns:
    - set: A set containing unique task names.

    Notes:
    - This function depends on another function `extract_file_info` to parse the filename and extract the task name.
    - 'N/A' will be added to the set if a task name cannot be determined for a given table.
    """
    tasks = set()
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))
    return tasks

import plotly.express as px

def generate_divergent_colors(n):
    """
    Generates and returns a list of divergent colors using the cyclical HSV color scheme from Plotly Express.

    Parameters:
    - n (int): The number of unique colors needed.

    Returns:
    - list: A list containing 'n' divergent colors.

    Notes:
    - The function uses the HSV color scheme available in Plotly Express as the base.
    - The step size for picking colors from the HSV scheme is adjusted based on the value of 'n' to ensure that colors are divergent.
    """
    turbo_colors = px.colors.qualitative.Plotly
    step = max(1, len(turbo_colors) // n)
    colors = [turbo_colors[i % len(turbo_colors)] for i in range(0, n * step, step)]
    return colors[:n]

def generate_figure(all_tables, repetition_times, signal, output_dir):
    """
    Generate Plotly figures to visualize a specified physiological signal across multiple tasks and subjects.
    
    Parameters:
    - all_tables (list): List of file paths to the .tsv files containing the physiological data.
    - repetition_times (list): List of repetition times corresponding to each table.
    - signal (str): The physiological signal of interest to be plotted (e.g., 'framewise_displacement', 'std_dvars').
    - output_dir (str): Directory where the generated figures will be saved.
    
    Notes:
    - The function generates two types of figures: one for each unique task and one for all tasks.
    - Each figure has dropdown options to toggle visibility of data for each subject.
    - The color of the traces in the 'all tasks' figure depends on the task.
    - The function saves the figures in HTML format in the specified output directory.
        """
    # Initialize task_colors dictionary and tasks set
    task_colors = {}
    tasks = set()

    # Create a figure for all curves
    fig_all = go.Figure()

    # Extract unique tasks from all tables
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))

    # Loop through each unique task
    for task in tasks:
        fig_task = go.Figure()
        subject_data = {}
        
        # Loop through all tables and their respective repetition times
        for table, repetition_time in zip(all_tables, repetition_times):
            df = pd.read_csv(table, sep='\t')
            file_info = extract_file_info(os.path.basename(table).split('.')[0])

            # Check if the signal column exists in the data frame
            if signal in df.columns:
                subject_name = os.path.basename(table).split('_')[0]
                if subject_name not in subject_data:
                    subject_data[subject_name] = []
                
                # Extract signal values and time indices
                signal_values = df[signal]
                time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)

                subject_data[subject_name].append((table, time_indices, signal_values))

        # Initialize visibility list for fig_task and fig_all
        visibility_lists = []
        visibility_all_lists = []
        
        # Loop through all subjects and their data
        for subject, data_list in subject_data.items():
            visibility = [False] * len(all_tables)
            visibility_all = [False] * len(all_tables)
            
            for current_table, time_indices, signal_values in data_list:
                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])
                current_subject_name = os.path.basename(current_table).split('_')[0]
                current_session = os.path.basename(current_table).split('_')[1]
                current_task_name = file_info.get('task')
                current_run = os.path.basename(current_table).split('_')[3]

                # Create custom legend for the plot
                custom_legend = f"{current_subject_name} - {current_session} - {current_task_name} - {current_run}"

                # Generate unique colors for tasks if not already done
                unique_task_count = len(tasks)
                unique_colors = generate_divergent_colors(unique_task_count)
                if current_task_name not in task_colors:
                    task_colors = {task: color for task, color in zip(tasks, unique_colors)}
                
                color = task_colors.get(current_task_name)

                # Add trace to fig_task if task matches
                if file_info.get('task') == task:
                    fig_task.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
                
                # Add trace to fig_all
                fig_all.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend, line=dict(color=color)))

                # Update visibility list for fig_task and fig_all
                current_trace_index_task = len(fig_task.data) - 1
                current_trace_index_all = len(fig_all.data) - 1

                if current_trace_index_task >= len(visibility):
                    extend_length = (current_trace_index_task - len(visibility) + 1)
                    visibility.extend([False] * extend_length)
                visibility[current_trace_index_task] = True

                if current_trace_index_all >= len(visibility_all):
                    extend_length = (current_trace_index_all - len(visibility_all) + 1)
                    visibility_all.extend([False] * extend_length)
                visibility_all[current_trace_index_all] = True

            # Append updated visibility lists for fig_task and fig_all
            visibility_lists.append(visibility)
            visibility_all_lists.append(visibility_all)

        # Set y-axis title based on signal
        yaxis_title = signal
        if signal == 'framewise_displacement':
            yaxis_title = 'FD (mm)'
        elif signal == 'std_dvars':
            yaxis_title = 'Standardized DVARS'

        # Update layout for fig_task
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

        # Update layout for fig_all
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

        # Generate dropdown menus for fig_task and fig_all
        dropdown_buttons_tasks = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_task.data)}, {'title': f'{signal} for All Subjects in task {task}', 'showlegend': True}])]
        dropdown_buttons_all = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_all.data)}, {'title': f'{signal} for All Subjects in all tasks', 'showlegend': True}])]

        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_tasks.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{signal} for {subject} in task {task}', 'showlegend': True}]))
        
        fig_task.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_tasks, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        
        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_all_lists[i]}, {'title': f'{signal} for {subject} in all tasks', 'showlegend': True}]))
        
        fig_all.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_all, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the fig_task as HTML
        fig_name = f"desc-tasks{signal}_signal_for_task-{task}.html"
        fig_task.write_html(os.path.join(output_dir, fig_name))
                
        # Save the fig_all as HTML
        fig_all_name = f"desc-all{signal}_signal_for_all_tasks.html"
        fig_all.write_html(os.path.join(output_dir, fig_all_name))



def generate_figures_motion(all_tables, repetition_times, signals, output_dir):
    """
    The generate_figures_motion function in the code is designed to generate Plotly figures for motion parameters in neuroimaging data. The function takes in four arguments:

    all_tables: A list of file paths to tables containing motion parameters.
    repetition_times: A list of repetition times associated with the tables.
    signals: A list of signals (motion parameters like 'rot_z', 'trans_z') to plot.
    output_dir: The directory where the figures will be saved.
    """
    # Initialize task set and global figure
    tasks = set()
    fig_all = go.Figure()

    # Color map for lines in plot
    color_map = ['darkred', 'black', 'grey']

    # Extract task info from all tables
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))

    # Loop through each task
    for task in tasks:
        fig_task = go.Figure()
        subject_data = {}

        # Load and prepare data for each table
        for table, repetition_time in zip(all_tables, repetition_times):
            df = pd.read_csv(table, sep='\t')
            file_info = extract_file_info(os.path.basename(table).split('.')[0])

            subject_name = os.path.basename(table).split('_')[0]
            if subject_name not in subject_data:
                subject_data[subject_name] = []

            time_indices = np.arange(0, len(df) * repetition_time, repetition_time)
            subject_data[subject_name].append((table, time_indices, df))

        # Lists to manage visibility of traces in the dropdown
        visibility_lists = []
        visibility_all_lists = []

        # Loop through each subject's data
        for subject, data_list in subject_data.items():
            visibility = [False] * len(all_tables) * len(signals)
            visibility_all = [False] * len(all_tables) * len(signals)

            # Plot the signals
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

            # Update visibility lists
            visibility_lists.append(visibility)
            visibility_all_lists.append(visibility_all)

        # Set display labels based on signals
        if signal == 'rot_z':
            display_signal = 'Rotation'
        elif signal == 'trans_z':
            display_signal = 'Translation'

        # Update layout for individual task figure
        fig_task.update_layout(
            hoverlabel_namelength=-1,
            title={
                'text': f'{display_signal} for task {task}',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'{display_signal}', 
            autosize=True
        )

        # Update layout for global figure
        fig_all.update_layout(
            hoverlabel_namelength=-1,
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

        # Create dropdown menus
        dropdown_buttons_tasks = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_task.data)}, {'title': f'{display_signal} for all Subjects in task {task}', 'showlegend': True}])]
        dropdown_buttons_all = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_all.data)}, {'title': f'{display_signal} for all Subjects in all tasks', 'showlegend': True}])]

        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_tasks.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'{display_signal} for {subject} in task {task}', 'showlegend': True}]))
        
        fig_task.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_tasks, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
        
        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_all_lists[i]}, {'title': f'{display_signal} for {subject} in all tasks', 'showlegend': True}]))
        
        fig_all.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_all, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

        # Save figures as HTML
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig_name = f"desc-tasks{display_signal} _for_task-{task}.html"
        fig_task.write_html(os.path.join(output_dir, fig_name))

    # Save global figure as HTML
    fig_all_name = f"desc-all{display_signal} _for_all_tasks.html"
    fig_all.write_html(os.path.join(output_dir, fig_all_name))



def display_outliers(all_tables, repetition_times, output_dir, fd_threshold=0.5, dvars_threshold=1.5):
    """The display_outliers function visualizes outlier motion parameters in neuroimaging data using Plotly. Here are the main functionalities of this function:
    Initialization: Initializes sets for tasks (tasks), a global figure (fig_all), and task colors (task_colors).
    Identifying Tasks: The function iterates through each table to identify unique tasks and populates the tasks set.
    Collecting Subject Data: For each table, it reads the data, extracts motion parameters, and identifies outliers based on FD (Framewise Displacement) and DVARS (Derivative of RMS Variance over voxels) thresholds.
    Plotting: Plots these outliers in fig_all, which aggregates all tasks.
    Dropdown Menu: Updates the layout and dropdown menus to filter by subject.
    Saving Figures: Saves the generated figure to an HTML file.
    """
    task_colors = {}  # Color map for each task
    tasks = set()  # To keep track of tasks
    fig_all = go.Figure()  # Global figure for all curves
    
    # Populate 'tasks' by going through all tables
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))
    
    # Iterate over each task
    for task in tasks:
        subject_data = {}  # To store subject-wise data

        # Read and prepare data for each table
        for table, repetition_time in zip(all_tables, repetition_times):
            df = pd.read_csv(table, sep='\t')
            file_info = extract_file_info(os.path.basename(table).split('.')[0])
            subject_name = os.path.basename(table).split('_')[0]

            if subject_name not in subject_data:
                subject_data[subject_name] = []

            if fd_threshold != 0.5 or dvars_threshold != 1.5:
                fd_col = 'framewise_displacement'
                dvars_col = 'std_dvars'
                outliers = np.where((df[fd_col] > fd_threshold) | (df[dvars_col] > dvars_threshold), 1, 0)
                signal_values = outliers
            else:
                motion_outliers = [col for col in df.columns if 'motion_outlier' in col]
                df = df[motion_outliers]
                outliers = df.sum(axis=1)
                signal_values = outliers

            time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)

            subject_data[subject_name].append((table, time_indices, signal_values))

        visibility_lists = []
        
        for subject, data_list in subject_data.items():
            visibility = [False] * len(all_tables)
            

            for current_table, time_indices, signal_values in data_list:
                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])
                current_subject_name = os.path.basename(current_table).split('_')[0]
                current_session = os.path.basename(current_table).split('_')[1]
                current_task_name = file_info.get('task')
                current_run = os.path.basename(current_table).split('_')[3]

                custom_legend = f"{current_subject_name} - {current_session} - {current_task_name} - {current_run}"

                unique_task_count = len(tasks)
                unique_colors = generate_divergent_colors(unique_task_count)

                if current_task_name not in task_colors:
                    task_colors = {task: color for task, color in zip(tasks, unique_colors)}
                
                color = task_colors.get(current_task_name) 

                file_info = extract_file_info(os.path.basename(current_table).split('.')[0])

                filtered_indices = np.where(signal_values == 1)[0]
                filtered_time_indices = time_indices[filtered_indices]
                filtered_signal_values = signal_values[filtered_indices]

                jitter_amount = 0.1
                jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(filtered_time_indices))
                jittered_signal_value = filtered_signal_values + jitter
                fig_all.add_trace(go.Scatter(x=filtered_time_indices, y=filtered_signal_values, mode='markers', name=custom_legend, marker=dict(color=color)))

                current_trace_index_all = len(fig_all.data) - 1

                if current_trace_index_all >= len(visibility):
                    extend_length = (current_trace_index_all - len(visibility) + 1)
                    visibility.extend([False] * extend_length)
                visibility[current_trace_index_all] = True

            visibility_lists.append(visibility)

        yaxis_title = outliers
        fig_all.update_yaxes(range=[0.5, 1.5])



        fig_all.update_layout(
            hoverlabel_namelength=-1,
            title={
                'text': f'outliers for all tasks',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
            xaxis_title='Time (seconds)', 
            yaxis_title=f'Outliers', 
            autosize=True
        )


        # Create dropdown menu for global figure
        dropdown_buttons_all = [dict(label="All subjects", method='update', args=[{'visible': [True]*len(fig_all.data)}, {'title': f'Outliers for All Subjects in all tasks', 'showlegend': True}])]
        
        for i, (subject, _) in enumerate(subject_data.items()):
            dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_lists[i]}, {'title': f'Outliers for {subject} in all tasks', 'showlegend': True}]))

        fig_all.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons_all, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save global figure as HTML
        fig_all_name = f"desc-outliers_signal_for_all_tasks.html"
        fig_all.write_html(os.path.join(output_dir, fig_all_name))
    

        

def generate_report_with_plots(
    output_dir,      # Directory to save the generated report
    run_uuid,        # Unique ID for the current run
    reportlets_dir,  # Directory where reportlets (small report chunks) are stored
    bootstrap_file,  # File for Bootstrap (presumably a CSS/JS framework)
    metadata,        # Metadata to be included in the report
    plugin_meta,     # Plugin metadata, probably configuration details for some kind of plugin
    tasks,           # Task-related data
    **entities       # Additional parameters
):
    out_filenames = []  # List to store output filenames

    # Create a Report object, which will handle the actual report generation
    robj = Report(
        output_dir,
        run_uuid,
        reportlets_dir=reportlets_dir,
        bootstrap_file=bootstrap_file,
        metadata=metadata,
        plugin_meta=plugin_meta,
        **entities
    )
    
    # Generate the report using the Report object
    robj.generate_report()

    # Renaming the generated report file
    original_filename = robj.out_filename.absolute()  # Get the absolute path of the generated report
    new_filename = os.path.join(output_dir, f"report_fmriprep_group.html")  # Create a new filename
    os.rename(original_filename, new_filename)  # Rename the file

    # Append the new filename to the list of output filenames
    out_filenames.append(new_filename)

    return out_filenames  # Return the list of output filenames

