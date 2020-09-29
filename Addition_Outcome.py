import pandas as pd
import numpy as np
import DataCleaning as dc
import os

outpath=r"C:\Users\prach\Downloads"

df=dc.ScalingMoistureValues()
df["Output"] = np.where(df['Moisture 1']<0.5, 1, 0)
df.to_csv(os.path.join(outpath, "AddedOutcome.csv"))