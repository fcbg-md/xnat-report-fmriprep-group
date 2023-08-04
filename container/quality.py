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
from nibabel.freesurfer.mghformat import load
from nireports.assembler.report import Report


def get_bids_data(data_path):
    deriv_path = os.path.join(data_path, "derivatives/fmriprep")

    layout = BIDSLayout(data_path, derivatives=True)
    lay=BIDSLayout(data_path)

    all_tables = layout.get(extension='.tsv', suffix='timeseries', scope='derivatives', return_type='filename')
    information_files = lay.get(extension='.json', suffix='bold', return_type='sucject')  

    subject_names = []

    for table in all_tables:
        match = re.search(r'sub-([a-zA-Z0-9]+)/ses-([a-zA-Z0-9]+)/.*run-([a-zA-Z0-9]+)', table)
        if match:
            subject_name = f"sub-{match.group(1)}_ses-{match.group(2)}_run{match.group(3)}"
            subject_names.append(subject_name)

    repetition_times = []

    for file_path in information_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        repetition_times.append(data.get('RepetitionTime'))

    return all_tables, repetition_times



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
    subject_names = [os.path.basename(table).split('.')[0] for table in all_tables]

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
    output_dir = os.path.join(output_dir, "report", "reportlets", "figures")
    
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
    )  # Cette parenthèse fermante est nécessaire

    # Specify the directory to save the file
    output_dir = os.path.join(output_dir, "report", "reportlets", "figures")
    
    # Check if the directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure to an HTML file
    fig_name = f"desc-{signal}_signal_for_all_subjects.html"
    fig.write_html(os.path.join(output_dir, fig_name))


# from sklearn.decomposition import PCA

# def perform_pca(all_files, output_dir):

#     output_dir = os.path.join(output_dir, "report", "reportlets", "figures")

#     means = []
#     for file_path in all_files:
#         df = pd.read_csv(file_path, sep='\t')
#         df_mean = df[['framewise_displacement', 'dvars', 'std_dvars', 'rmsd']].mean()
#         df_mean.fillna(df_mean.mean(), inplace=True)
#         means.append(df_mean)

#     df_all_means = pd.concat(means, axis=1)

#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(df_all_means)

#         # Ensure the directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Assuming pca_result is an array containing your PCA data
#     fig1 = go.Figure(data=go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers'))
#     fig1.update_layout(title='PCA - Principal Component Analysis', xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')

#     # Now you can write to the directory
#     fig1.write_html(os.path.join(output_dir, 'pca_plot.html'))

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

