# Author        : Simon Bross
# Date          : January 17, 2024
# Python version: 3.10

import pandas as pd
import os


def get_data_df():
    """
    Reads in the entire ClaimBuster dataset (22501 sentences from
    crowdsourced.csv) and converts it into a pandas dataframe,
    thereby mapping the verdict numerical values (= class labels)
    to their respective string labels.
    @return: Pandas dataframe.
    """
    path = os.path.join("data", "crowdsourced.csv")
    data_df = pd.read_csv(path)
    # labels in the dataset are numeric (-1, 0, 1),
    # map them to their respective representation (NFS, UFS, CFS)
    mapping = {-1: "NFS", 0: "UFS", 1: "CFS"}
    data_df["Verdict"] = data_df["Verdict"].map(mapping)
    return data_df
