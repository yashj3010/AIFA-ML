import pandas as pd
from sklearn import preprocessing
import os


# Function to scale all moisture values between maximum and minimum hydration levels

def RangeScaling(val, min_val, max_val ):
    if val<min_val:
        return min_val
    elif val>max_val:
        return max_val
    else:
        return val

# Function to normalize data: replaces value, val by (val-min/max-min) : where max and min are the maximum and minimum column values respectively.

def NormalizingValues(inpath, outpath, min_val, max_val):                       #Scales all values between 0-1

    df = pd.read_csv(inpath, index_col = False)

    df["Moisture 1"] = df.apply(lambda row: RangeScaling(row['Moisture 1'], min_val, max_val), axis=1 )
    df["Moisture 2"] = df.apply(lambda row: RangeScaling(row['Moisture 2'], min_val, max_val), axis=1)

    print("\n----------- Minimum -----------\n")
    print(df.min())
    
    print("\n----------- Maximum -----------\n")
    print(df.max())

    min_max_scaler = preprocessing.MinMaxScaler()
    df["Moisture 1"] = min_max_scaler.fit_transform(df[["Moisture 1"]])
    df["Moisture 2"] = min_max_scaler.fit_transform(df[["Moisture 2"]])
    df["Light"] = min_max_scaler.fit_transform(df[["Light"]])
    df["Temp"] = min_max_scaler.fit_transform(df[["Temp"]])
    df["Humidity"] = min_max_scaler.fit_transform(df[["Humidity"]])


    df.to_csv(os.path.join(outpath,"NormalizedData.csv"), index = False)

    return df

