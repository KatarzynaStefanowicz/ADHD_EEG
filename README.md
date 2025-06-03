# EEG-Based ADHD Detection Using AI

This project aims to build a software tool that can help detect ADHD using long-term EEG data and machine learning. We're experimenting with different models to find out which approach works best for identifying ADHD-related patterns in EEG signals.

The project uses data from this Kaggle dataset:
https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data

**Project Goals**
   - Use AI to detect signs of ADHD from EEG recordings

  - Compare different machine learning and deep learning models

  - Build the system in a modular way, so we can easily switch between:

    - Different EEG channels

    - Various model architectures

  - Evaluate which channels and model setups give the most accurate results
***

**Structure (Example)**
```
├── data/                 # EEG datasets and preprocessing scripts
├── models/               # Model architectures
├── experiments/          # Experiment configurations and results
├── utils/                # Utility functions
├── main.py               # Entry point for training
├── adhd_eeg_env.yaml     # Enviroment file
└── README.md             # Project overview
```
---
# How to Run It
Before running the project, we recommend using a virtual environment to keep dependencies organized and avoid conflicts. You can set it up using the .yaml file provided in the repository: adhd_eeg_env.yaml.

**Set Up the Virtual Environment (Anaconda)**
If you don’t already have Anaconda installed, you can download it here:
https://www.anaconda.com/products/distribution

Then follow these steps:

**Option 1: Using the Command Line (Anaconda Prompt or terminal)**
- Download or clone the repo:

```
git clone https://github.com/_your-username_/adhd-eeg-ai.git
cd adhd-eeg-ai
```

- Create the environment from the YAML file:

```
conda env create -f adhd_eeg_env.yaml
```

- Activate the environment:

```
conda activate adhd-eeg-env
```

**Option 2: Using Anaconda Navigator**
1. Open Anaconda Navigator

2. Go to the **"Environments"** tab on the left

3. Click the **"Import"** button (bottom left)

4. In the dialog:

   - Name the environment (e.g., adhd-eeg-env)

   - For YAML file, click **"Browse"** and select **adhd_eeg_env.yaml** from your local repo folder

5. Click **"Import"** - Anaconda will create the environment with the correct dependencies

6. Once it’s done, go to the **"Enviroments"** tab and select the new environment
   
7. Then go to the **"Home"** tab and launch a VSCode or Jupyter from there
