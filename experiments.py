# Author        : Simon Bross
# Date          : January 27, 2024
# Python version: 3.6.13

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from data.get_data import get_data_df
from features import FeatureExtractor
from classifiers import get_classifier


def run_experiment(method, features="all", error_analysis=True):
    """
    Performs a classification experiment on the ClaimBuster dataset using
    stratified 5-fold cross-validation, given an ML method specified in the
    classifiers.py script and a set of (linguistic) features that represent
    the data. For evaluation, a classification report is generated after every
    fold and stored according to the function parameters (cf. store_report).
    If desired, a confusion matrix will be generated and stored for every fold,
    as well as the wrongly labeled sentences with their predicted labels and
    gold labels. This data can be used for a subsequent error analysis.
    @param method: Name of the classifier from classifiers.py to use, as str.
    @param features: Features to extract from the data. Defaults to 'all'.
    Alternatively, a subset of the available features can be passed via
    a list of strings (cf. features.py).
    @param error_analysis: Whether to store the confusion matrix (as pdf) as
    well as the wrongly labeled sentences along with their predicted and gold
    labels as a pandas dataframe, defaults to False.
    """
    # Retrieve data
    X = get_data_df()["Text"].tolist()
    y = get_data_df()["Verdict"].tolist()
    # use stratified K fold to ensure a well-balanced class distribution
    skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold number: {i}\n")
        X_train_plain = [X[p] for p in train_index]
        y_train = [y[p] for p in train_index]
        X_test_plain = [X[p] for p in test_index]
        y_test = [y[p] for p in test_index]
        # retrieve features
        extractor = FeatureExtractor(X_train_plain, X_test_plain)
        X_train, X_test = extractor.get_features(select=features)
        model = get_classifier(method)
        print(f"Fitting {model}")
        model.fit(X_train, y_train)
        print(f"Testing and evaluating {model}")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        # store report
        store_report(report, i, method, features)
        if error_analysis:
            # get indices of wrongly predicted labels
            wrong_pred = [
                k for k in range(len(y_pred)) if y_pred[k] != y_test[k]
            ]
            # get test sentences with their wrongly predicted and gold labels
            sent_pred_gold = [
                    {
                        "sentence": X_test_plain[n],
                        "predicted": y_pred[n],
                        "gold": y_test[n]
                    } for n in wrong_pred
            ]
            df = pd.DataFrame(sent_pred_gold)
            store_df(df, i, method, features)
            # get confusion matrix and store them
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=model.classes_
            )
            store_plot(disp, i, method, features)


def store_plot(plot, fold_num, method, features):
    """
    Saves the confusion matrix as a .pdf file in the 'error_analysis/plots'
    directory and respective subdirectory (named after ML method employed).
    The plot to be stored is named according to the fold number and
    the features used.
    @param plot: ConfusionMatrixDisplay instance.
    @param fold_num: Fold number of k-fold cross validation as int, indicating
    which iteration of the process produced the report.
    @param method: The name of the classifier used, specifying the ML model
    employed for the experiment, as str.
    @param features: The features that were extracted from the data, either all
    available (='all') or a subset as a list of strings.
    """
    if not os.path.isdir("error_analysis"):
        os.mkdir("error_analysis")
    if not os.path.isdir(os.path.join("error_analysis", "plots")):
        os.mkdir(os.path.join("error_analysis", "plots"))
    if not os.path.isdir(os.path.join("error_analysis", "plots", method)):
        os.mkdir(os.path.join("error_analysis", "plots", method))
    if features != "all":
        file_name = f"fold_{fold_num}_{'_'.join(features)}.pdf"
    else:
        file_name = f"fold_{fold_num}_all.pdf"
    file_path = os.path.join("error_analysis", "plots", method, file_name)
    plot.plot(values_format="d")
    plt.savefig(file_path)


def store_df(df, fold_num, method, features):
    """
    Saves the wrongly labeled sentences (with predicted and gold labels)
    for an experiment fold in a json file in the 'error_analysis/dfs'
    directory and respective subdirectory (named after ML method employed).
    The dataframe to be stored is named according to the fold number and
    the features used.
    @param df: Dataframe to be stored.
    @param fold_num: Fold number of k-fold cross validation as int, indicating
    which iteration of the process produced the report.
    @param method: The name of the classifier used, specifying the ML model
    employed for the experiment, as str.
    @param features: The features that were extracted from the data, either all
    available (='all') or a subset as a list of strings.
    """
    if not os.path.isdir("error_analysis"):
        os.mkdir("error_analysis")
    if not os.path.isdir(os.path.join("error_analysis", "dfs")):
        os.mkdir(os.path.join("error_analysis", "dfs"))
    if not os.path.isdir(os.path.join("error_analysis", "dfs", method)):
        os.mkdir(os.path.join("error_analysis", "dfs", method))
    if features != "all":
        file_name = f"df_fold_{fold_num}_{'_'.join(features)}.json"
    else:
        file_name = f"df_fold_{fold_num}_all.json"
    file_path = os.path.join("error_analysis", "dfs", method, file_name)
    df.to_json(file_path)


def store_report(report, fold_num, method, features):
    """
    Saves the classification report for an experiment fold in a text file
    in the 'reports' directory and respective subdirectory (named
    after ML method employed). The report to be stored is named according to
    the fold number and the features used.
    @param report: Classification report to be stored, as str.
    @param fold_num: Fold number of k-fold cross validation as int, indicating
    which iteration of the process produced the report.
    @param method: The name of the classifier used, specifying the ML model
    employed for the experiment, as str.
    @param features: The features that were extracted from the data, either all
    available (='all') or a subset as a list of strings.
    """
    if not os.path.isdir("reports"):
        os.mkdir("reports")
    # check for subdirectory
    if not os.path.isdir(os.path.join("reports", method)):
        os.mkdir(os.path.join("reports", method))
    if features != "all":
        file_name = f"fold_{fold_num}_{'_'.join(features)}.txt"
    else:
        file_name = f"fold_{fold_num}_all.txt"
    file_path = os.path.join("reports", method, file_name)
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(report)
