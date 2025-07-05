EEG Frequency Analysis using Welch Method

This project contains a Jupyter notebook for analyzing EEG data using the Welch method for power spectral density estimation. The analysis focuses on transforming raw EEG time-series data into the frequency domain and classifying ADHD versus control subjects using classical machine learning methods.

Features

- Bandpass filtering of EEG signals between 1 and 30 Hz
- Power Spectral Density estimation using the Welch method
- Data augmentation for class balancing
- Feature extraction and labeling
- Classification using:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - K-Nearest Neighbors
- Model evaluation with classification reports

Dataset

The notebook uses EEG data stored in a CSV file located at data/adhdata.csv. Each row represents an EEG epoch across 19 standard 10-20 system channels with labels for ADHD diagnosis.

Requirements

To run the code, the following Python libraries are required:
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn

File Structure

WelchMethode_FrequencyDomainV5.ipynb   - Main analysis notebook
data/adhdata.csv                       - EEG dataset (not included here)

Usage

1. Clone or download this repository
2. Place the EEG CSV file into the data directory
3. Open and run WelchMethode_FrequencyDomainV5.ipynb in Jupyter Notebook

Output

The notebook produces:
- Visualizations of filtered signals and power spectral densities
- Classification accuracy and performance reports

License

This project is intended for academic and research use only.
