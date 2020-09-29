import pandas as pd
from sklearn import preprocessing
import TimeStampConverter as tsc
import DataReduction as dr
import os


min_val = 135                                       # sensor value for maximum hydration
max_val = 600                                       # sensor value for minimum hydration
inpath = r"C:\Users\prach\Downloads\Log Data\merge\RawDataSample.xlsx"
outpath = r"C:\Users\prach\Downloads"


# Function to scale all values between maximum and minimum hydration levels

def RangeScaling(val):
    if val<min_val:
        return min_val
    elif val>max_val:
        return max_val
    else:
        return val


def ScalingMoistureValues():                       #Scales all values between 0-1

    df = dr.meanobservation(inpath, outpath)
    df["Moisture 1"] = df.apply(lambda row: RangeScaling(row['Moisture 1']), axis=1 )
    min_max_scaler = preprocessing.MinMaxScaler()
    df["Moisture 1"] = min_max_scaler.fit_transform(df[["Moisture 1"]])
    df.to_csv(os.path.join(outpath,"FullyScaled.csv"))
    return df

if __name__ == "__main__":
    ScalingMoistureValues()