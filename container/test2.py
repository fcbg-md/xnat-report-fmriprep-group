from bids import BIDSLayout
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from quality import get_bids_data, generate_figure, generate_report_with_plots, generate_figure2, display_motion_outliers

data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
data_path = "/mnt/extra/data/share/bids_fmriprep_10subj/"
output_dir = "/home/axel/report-fmriprep/xnat-report-fmriprep-group/out_test"
#output_dir = os.path.join(data_path, "report")
reportlets_dir = os.path.join(output_dir, "/reportlets/figures")
all_tables, repetition_times = get_bids_data(data_path)

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

