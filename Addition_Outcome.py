import pandas as pd
import numpy as np
import DataCleaning as dc
import os

outpath=r"RawDataSample.xlsx"

df=dc.ScalingMoistureValues()
df["Output"] = np.where(df['Moisture 1']<0.5, 1, 0)
df.to_csv(os.path.join(outpath, "AddedOutcome.csv"))