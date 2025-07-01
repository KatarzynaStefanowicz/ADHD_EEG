EEG Data Augmentation and Preprocessing Toolkit

This repository folder contains Jupyter notebooks for augmenting, visualizing, and preprocessing EEG datasets, designed to improve the robustness and performance of machine learning models in EEG-based tasks.

Contents
1. EEG_Augmentation_DataMixing.ipynb
Description:

Demonstrates data augmentation through mixing of EEG samples.

Combines samples within or across classes to increase dataset diversity.

Usage:

Run this notebook to generate new mixed EEG samples.

2. EEG_Augmentation_NoiseInjection.ipynb
Description:

Adds controlled noise to EEG signals to improve model robustness.

Supports Gaussian noise, salt-and-pepper noise, or other customizable noise types.

Usage:

Run this notebook to apply noise-based augmentation to your dataset.

Parameters such as noise level and type can be adjusted within the notebook.

3. EEG_Augmentation_Visualization.ipynb
Description:

Provides visualization tools for comparing raw and augmented EEG signals.

Includes plots of time-series data, statistical comparisons, and signal properties.

Usage:

Run this notebook after applying augmentation to inspect results.

Helps validate that augmentations maintain signal integrity.

4. id_change.ipynb
Description:

Utility notebook for modifying or re-encoding participant or trial IDs in EEG datasets.

Useful for anonymization or reorganization of data identifiers.

Usage:

Load your dataset into the notebook and apply ID transformations as needed.

