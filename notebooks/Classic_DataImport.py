import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"data\adhdata.csv"
data = pd.read_csv(file_path)

data.head()
data.info()

sns.countplot(x=data['Class'], data=data)

patient = data[data.ID == 'v10p']

channel_names = data.columns.tolist()
channel_names.remove('Class')
channel_names.remove('ID')

# Define sampling frequency
sfreq = 128  # Sampling frequency in Hz

# Create MNE info object
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

# Display the MNE info object
print(info)

# Set the montage using standard 10-20 system
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Display the updated MNE info object with montage
print(info)

# Convert DataFrame to NumPy array
eeg_adhd_data = patient.drop(['Class','ID'], axis=1)
eeg_adhd_data = eeg_adhd_data.values.T  # Transpose to have channels as rows and samples as columns

# Create RawArray object with EEG data and MNE info
raw_adhd = mne.io.RawArray(eeg_adhd_data, info)

# Display the Raw object
print(raw_adhd)

# Plot raw EEG data
mne.set_config('MNE_BROWSE_RAW_SIZE','16,8')  
raw_adhd.plot(n_channels=len(channel_names), scalings='auto', title='Raw EEG Data')

# Apply bandpass filter (4-40 Hz)
raw_adhd_filtered = raw_adhd.copy().filter(l_freq=4, h_freq=40, method='fir', verbose=False)

# Plot PSD before and after filtering
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6))

# PSD Before Filtering
ax1 = fig.add_subplot(2, 1, 1)
raw_adhd.plot_psd(fmax=60, ax=ax1)  # Replace compute_psd with plot_psd
ax1.set_title('PSD Before Filtering')

# PSD After Filtering
ax2 = fig.add_subplot(2, 1, 2)
raw_adhd_filtered.plot_psd(fmax=60, ax=ax2)  # Replace compute_psd with plot_psd
ax2.set_title('PSD After Filtering')

plt.tight_layout()
plt.show()

raw_adhd_filtered.plot(scalings='auto')

from mne.preprocessing import ICA

# Initialize ICA
ica_adhd = ICA(n_components=19, random_state=42)

# Fit ICA to the preprocessed data
ica_adhd.fit(raw_adhd_filtered)

# Identify components related to ocular artifacts
eog_inds, scores = ica_adhd.find_bads_eog(raw_adhd_filtered, ch_name=['Fp1','Fp2','F7','F8'], threshold=3)
print(eog_inds)

# Plot ICA components in topographic maps
ica_adhd.plot_components()

# Plot ICA components in waveform
ica_adhd.plot_sources(raw_adhd_filtered)
plt.set_cmap('viridis')  # Change 'viridis' to your desired colorscale
plt.show()

# Exclude identified components from the ICA decomposition
ica_adhd.exclude = [6,10,14,16]

# Apply ICA to remove ocular artifacts
cleaned_raw_adhd = raw_adhd_filtered.copy()
cleaned_eeg_adhd = ica_adhd.apply(cleaned_raw_adhd)

cleaned_eeg_adhd.plot(scalings='auto')

eeg_adhd = cleaned_eeg_adhd.get_data()
eeg_adhd_df = pd.DataFrame(eeg_adhd.T, columns = channel_names, index=None)

eeg_adhd_df.head()