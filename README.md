# Leveraging Machine Learning for Automated Detection of Check-Worthy  Factual Claims

## Final Project in Argument Mining by Simon Bross (809648)

This project focuses on classifying sentences into three categories: Check-worthy Factual Sentences (CFS), Unimportant Factual Sentences (UFS), and Non-Factual Sentences (NFS). Employing a range of (linguistic) features, including sentiment, subjectivity, modality, mood, embeddings, and Part-of-Speech tags, the objective is to build models for sentence classification. A set of classifiers, including a basic neural network, is used to train and test models using varying feature combinations. 

## 1. Project Structure & Components

```
Project Directory
|-- classifiers.py
|-- experiments.py
|-- features.py
|-- main.py
|-- README.md
|-- requirements.txt
|
|-- data
|
|-- error_analysis
|   |-- dfs
|   └-- plots
|
└── reports
```

### 1.1 classifiers.py
Initializes the classifiers needed for the experiments. 

### 1.2 experiments.py

Provides the functionalities to run the experiments, including the storing of the evaluation reports and confusion matrices. 

### 1.3 features.py

Defines the FeatureExtractor class that extracts the features from textual data needed for the training and test data. 

### 1.4 main.py

Main script from which the experiments are run. 

### 1.5 requirements.txt
Lists all necessary dependencies for the project. Dependencies can be installed from the terminal using the following command: 
```
$ pip install -r requirements.txt 
```

### 1.6 'Data' Directory

Contains a script to retrieve the relevant data from the ClaimBuster dataset (crowdsourced.csv) and the original ReadMe file from it. 

### 1.7 'Error Analysis' Directory

Contains the subfolders 'dfs' and 'plots', in which the wrongly classified instances are saved in a dataframe together with the original sentences ('dfs') and the confusion matrices ('plots') are stored for every setting of fold number, feature combination, and classifier. 

### 1.8 'Reports' Directory

Contains the evaluation reports for every setting of fold number, feature combination, and classifier.

### 2. Python 

Due to incompatibility problems, this project uses the outdated Python version 3.6 for experimentation purposes.

### 3. Data

The 2020 version of the ClaimBuster dataset that contains labeled (NFS, CFS, UFS) sentences from presidential debates in the U.S. can be downloaded in its entirety under the following link: 

```https://zenodo.org/records/3836810#.YFqIX69KgUE```