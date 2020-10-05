import pandas as pd
import os

### Fuction to convert timestamp from object datatype to datetime64[ns]

def TimeStampFormatter(inpath, outpath):

        df = pd.read_csv(inpath, index_col= False)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%H:%M") #Edit format according to the need currently output in HH:MM form
        df["TimeStamp"] = pd.DatetimeIndex(df["TimeStamp"]).time
        df.to_csv(os.path.join(outpath, "TimeStampFormatted.csv"), index= False)
        return df




