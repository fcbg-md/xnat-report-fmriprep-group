from bids import BIDSLayout
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from quality import get_bids_data, generate_figure, generate_report_with_plots, generate_figure2

data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
output_dir = "/home/axel/report-fmriprep/xnat-report-fmriprep-group/out_test"
#output_dir = os.path.join(data_path, "report")
reportlets_dir = os.path.join(output_dir, "/reportlets/figures")
all_tables, repetition_times = get_bids_data(data_path)


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


    visibility_lists = []

    for subject, data_list in subjects_data.items():
        visibility = [False] * len(all_tables)  # Ensure visibility is initialized here for each subject

        for current_table, time_indices, signal_values in data_list:
            file_info = extract_file_info(os.path.basename(current_table).split('.')[0])

            # Create a custom legend using the extracted file info
            custom_legend = f"{subject}_ses-{file_info.get('ses', 'N/A')}_task-{file_info.get('task', 'N/A')}_run-{file_info.get('run', 'N/A')}"
            
            fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend))
            visibility[-1] = True  # set the latest trace as visible

        visibility_lists.append(visibility)

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


fig = go.Figure()

table = all_tables[105]

df = pd.read_csv(table, sep='\t')  # You forgot this line

if 'global_signal' in df.columns:
    framewise_displacement = df['framewise_displacement']
    
    # FFT and frequency computation
    fft_vals = np.fft.rfft(framewise_displacement)  # compute FFT
    fft_freq = np.fft.rfftfreq(len(framewise_displacement), d=1.)  # compute frequencies

    # Filter frequencies higher than 0.02
    fft_vals[fft_freq > 0.02] = 0

    # Inverse FFT to get filtered time-domain signal
    filtered_signal = np.fft.irfft(fft_vals)

    # Create a Plotly figure
    fig = go.Figure()

    # Add original and filtered signals to the figure
    fig.add_trace(go.Scatter(y=framewise_displacement, mode='lines', name='Original Signal'))
    #fig.add_trace(go.Scatter(y=framewise_displacement, mode='lines', name='Filtered Signal'))

    fig.update_layout(title='framewise_displacement Signal (Original and Filtered)', 
                      xaxis_title='Time', 
                      yaxis_title='framewise_displacement Signal', 
                      autosize=True)
    fig.show()

from scipy.signal import spectrogram

# Assume that 'global_signal' is your signal and 'fs' is the sampling ratefs
fs=0.02
frequencies, times, Sxx = spectrogram(framewise_displacement, fs, nperseg=3)


# Calculate mean PSD for each time point
mean_psd = np.mean(10 * np.log10(Sxx), axis=0)

# Calculate mean PSD for each time point
mean_psd = np.mean(10 * np.log10(Sxx), axis=0)
# Plot mean PSD over time
plt.plot(times, mean_psd)
plt.xlabel('Time [sec]')
plt.ylabel('Mean PSD [dB]')
plt.title('Mean Spectral Power Density over Time')
plt.show()
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

# Your code
normalized_mean_psd = (mean_psd - np.min(mean_psd)) / (np.max(mean_psd) - np.min(mean_psd))
enhanced_mean_psd = normalized_mean_psd ** 2
mean_enhanced_psd = np.mean(enhanced_mean_psd)
std_enhanced_psd = np.std(enhanced_mean_psd)

threshold = mean_enhanced_psd + 2*std_enhanced_psd
anomaly_indices = np.where(enhanced_mean_psd > threshold)[0]

# Group the consecutive indices
grouped_anomaly_indices = group_consecutives(anomaly_indices)


for group in grouped_anomaly_indices:
    print(enhanced_mean_psd[group])

# Visualize the data and anomalies
plt.figure(figsize=(10, 6))
plt.plot(times, enhanced_mean_psd, label='Enhanced Mean PSD')
for group in grouped_anomaly_indices:
    plt.plot(times[group], enhanced_mean_psd[group], 'ro', markersize=4, label='Anomalies')
plt.legend()
plt.title('Enhanced Mean PSD and Anomalies')
plt.xlabel('Time [sec]')
plt.ylabel('Enhanced Mean PSD')
plt.show()

len(times)

