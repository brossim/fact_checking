# Author        : Simon Bross
# Date          : January 27, 2024
# Python version: 3.6.13

# Class distribution: {NFS: 14685, CFS: 5413, UFS: 2403})

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_classifier(classifier):
    """
    Initializes and returns an instance of a classifier from scikit-learn
    based on the specified classifier name.
    Available classifiers:
    - 'baseline': Dummy classifier with a 'stratified' strategy.
    - 'k_near': K-Nearest Neighbors classifier.
    - 'grad_boost': Gradient Boosting Classifier.
    - 'random_forest': Random Forest Classifier with balanced class weights.
    - 'svm': Support Vector Machine classifier with balanced class weights.
    - 'mlp': Multi-layer Perceptron classifier with early stopping.
    @param classifier: The name of the classifier to retrieve, as str.
    @return: An instance of the specified classifier.
    @raise ValueError: If the specified classifier name is not valid.
    """
    if classifier == "baseline":
        baseline = DummyClassifier(strategy="stratified", random_state=123)
        return baseline
    elif classifier == "k_near":
        k_near = KNeighborsClassifier(weights="distance")
        return k_near
    elif classifier == "grad_boost":
        complement_nb = GradientBoostingClassifier(
            n_estimators=300, random_state=123)
        return complement_nb
    elif classifier == "random_forest":
        rfc = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=123
        )
        return rfc
    elif classifier == "svm":
        svm = SVC(random_state=123, class_weight='balanced')
        return svm
    elif classifier == "mlp":
        mlp = MLPClassifier(
            hidden_layer_sizes=1000, solver="sgd", learning_rate="adaptive",
            max_iter=500, learning_rate_init=0.001, random_state=123,
            early_stopping=True, verbose=True
        )
        return mlp
    else:
        raise ValueError(f"'{classifier}' is not a valid classifier")
