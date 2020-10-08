import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def NNInputFile(inpath, outpath, testsizeval, randomstateval, bool ):

    df = pd.read_csv(inpath, index_col = False)

    train, test = train_test_split(df, test_size = testsizeval, random_state = randomstateval, shuffle = bool)

    train_output = pd.DataFrame(columns=["Output"])
    train_output["Output"] = train["Output"]
    train_input = train.drop(['Output'], axis=1)

    test_output = pd.DataFrame(columns=["Output"])
    test_output["Output"] = test["Output"]
    test_input = test.drop(['Output'], axis=1)

    train_input.to_csv(os.path.join(outpath, "NNModelTraining" + "TrainInput.csv"))
    train_output.to_csv(os.path.join(outpath, "NNModelTraining" +  "TrainOutput.csv"))
    test_input.to_csv(os.path.join(outpath, "NNModelTraining" +  "TestInput.csv"))
    test_output.to_csv(os.path.join(outpath, "NNModelTraining" +  "TestOutput.csv"))

