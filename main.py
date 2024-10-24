# Author        : Simon Bross
# Date          : January 28, 2024
# Python version: 3.6.13

from experiments import run_experiment

# define feature combinations
feat_comb1 = "all"
feat_comb2 = ["embedding"]
feat_comb3 = [
    "question", "pos", "mood", "modality", "subjectivity",
    "sentiment"
]
feat_comb4 = [
    "question", "mood", "modality", "subjectivity",
    "sentiment"
]
all_combs = [feat_comb1, feat_comb2, feat_comb3, feat_comb4]

# run experiments for every combination and model
if __name__ == '__main__':
    for comb in all_combs:
        run_experiment("baseline", features=comb, error_analysis=True)
        run_experiment("k_near", features=comb, error_analysis=True)
        run_experiment("grad_boost", features=comb, error_analysis=True)
        run_experiment("random_forest", features=comb, error_analysis=True)
        run_experiment("svm", features=comb, error_analysis=True)
        run_experiment("mlp", features=comb, error_analysis=True)
