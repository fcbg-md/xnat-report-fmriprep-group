import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.signal import spectrogram
from bids import BIDSLayout

data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
deriv_path = "/mnt/extra/data/share/bids_fmriprep_10subj/derivatives/fmriprep/"

layout = BIDSLayout(data_path, derivatives=True)
lay=BIDSLayout(data_path)


all_tables=layout.get(extension='.tsv', suffix='timeseries', scope='derivatives', return_type='filename')
import re

subject_names = []

for table in all_tables:
    match = re.search(r'sub-([a-zA-Z0-9]+)/ses-([a-zA-Z0-9]+)/.*run-([a-zA-Z0-9]+)', table)
    if match:
        subject_name = f"sub-{match.group(1)}_ses-{match.group(2)}_run{match.group(3)}"
        subject_names.append(subject_name)

print(subject_names)
print(len(subject_names))


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

data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
deriv_path = "/mnt/extra/data/share/bids_fmriprep_10subj/derivatives/fmriprep/"

layout = BIDSLayout(data_path, derivatives=True)

fs=0.01

all_tables=layout.get(extension='.tsv', suffix='timeseries', scope='derivatives', return_type='filename')

fig = go.Figure()

# Create a list of file names for the dropdown options
file_names = [f"File {i+1}" for i in range(len(all_tables))]

fig = go.Figure()

# Create a list of visibility lists
visibility_lists = []

for i, table in enumerate(all_tables):
    df = pd.read_csv(table, sep='\t')

    if 'global_signal' in df.columns:
        global_signal = df['global_signal']

        frequencies, times, Sxx = spectrogram(global_signal, fs, nperseg=3)

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
        
        fig.add_trace(go.Scatter(y=global_signal, mode='lines', line_color='darkgray', name=subject_names[i]))

        # The last trace added should be visible for this file
        visibility.append(True)

        for j, group in enumerate(grouped_anomaly_indices):
            if group: 
                start_index = (group[0] * len(global_signal)) // len(times)
                end_index = (group[-1] * len(global_signal)) // len(times)
                start_index = max(min(start_index, len(global_signal)-1), 0)
                end_index = max(min(end_index, len(global_signal)-1), 0)
                anomalies_global = global_signal[start_index:end_index+1]
                time_indices = list(range(start_index, end_index+1))

                if len(group) > 1:
                    fig.add_trace(go.Scatter(x=time_indices, y=anomalies_global, mode='lines', line=dict(color=color_palette[j]), name=f'Anomaly {j+1} in {subject_names[i]}'))
                else:
                    fig.add_trace(go.Scatter(x=time_indices, y=anomalies_global, mode='markers', marker=dict(color=color_palette[j]), name=f'Anomaly {j+1} in {subject_names[i]}'))
                
                # The last trace added should be visible for this file
                visibility.append(True)

        # Add the visibility list for this file to the list of visibility lists
        visibility_lists.append(visibility)

fig.update_layout(title='Global Signals', 
                  xaxis_title='Time', 
                  yaxis_title='Global Signal', 
                  autosize=True)

# Create the dropdown menu
dropdown_buttons = []
for i, visibility in enumerate(visibility_lists):
    # Extend the visibility list to cover all traces
    visibility += [False] * (len(fig.data) - len(visibility))
    dropdown_buttons.append(dict(label=subject_names[i], method='update', 
                                 args=[{'visible': visibility}, 
                                       {'title': f'Global Signal for File {i+1}', 'showlegend': True}]))

# Add 'All Files' option
dropdown_buttons.append(dict(label='All group', method='update', 
                             args=[{'visible': [True]*len(fig.data)}, 
                                   {'title': 'Global Signals for All Files', 'showlegend': True}]))

fig.update_layout(updatemenus=[dict(active=len(dropdown_buttons)-1, buttons=dropdown_buttons)])

fig.show()
