import pandas as pd
import os


## Function to club all readings of a particular minute together.

def MeanObservation(inpath, outpath):

    df = pd.read_csv(inpath, index_col = False)
    reduceddf = df.groupby(pd.Grouper(key="TimeStamp")).mean()
    reduceddf.to_csv(os.path.join(outpath,"ReducedData.csv"))
    return reduceddf

