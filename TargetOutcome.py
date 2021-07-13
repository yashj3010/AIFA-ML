import pandas as pd
import numpy as np
import os

def DecisionMaker(inpath, outpath):
    df = pd.read_csv(inpath, index_col = False)
    df["Output"] = np.where(df['Moisture 1']<0.5, 1, 0)
    df.to_csv(os.path.join(outpath, "TargetOutcomeGenerated.csv"), index = False)