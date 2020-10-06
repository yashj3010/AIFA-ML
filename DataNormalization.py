import pandas as pd
from sklearn import preprocessing
import os
import csv 

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
    min_max_values = []

    df = pd.read_csv(inpath, index_col = False)

    df["Moisture 1"] = df.apply(lambda row: RangeScaling(row['Moisture 1'], min_val, max_val), axis=1 )
    df["Moisture 2"] = df.apply(lambda row: RangeScaling(row['Moisture 2'], min_val, max_val), axis=1)

    # print("\n----------- Minimum -----------\n")
        # print(df.min())
        # print(df["TimeStamp"].min())
        # print(df["Light"].min())
        # print(df["Moisture 1"].min())
        # print(df["Moisture 2"].min())
        # print(df["Temp"].min())
        # print(df["Humidity"].min())
        
    # print("\n----------- Maximum -----------\n")
        # print(df.max())
        # print(df["TimeStamp"].max())
        # print(df["Light"].max())
        # print(df["Moisture 1"].max())
        # print(df["Moisture 2"].max())
        # print(df["Temp"].max())
        # print(df["Humidity"].max())

    min_max_values.append(df["TimeStamp"].min())
    min_max_values.append(df["TimeStamp"].max())
    
    min_max_values.append(df["Light"].min())
    min_max_values.append(df["Light"].max())
    
    min_max_values.append(df["Moisture 1"].min())
    min_max_values.append(df["Moisture 1"].max())
    
    min_max_values.append(df["Moisture 2"].min())
    min_max_values.append(df["Moisture 2"].max())
    
    min_max_values.append(df["Temp"].min())
    min_max_values.append(df["Temp"].max())
    
    min_max_values.append(df["Humidity"].min())
    min_max_values.append(df["Humidity"].max())

    print(min_max_values)

    file = open(r'CSVs\Target\minMaxVals.csv', 'w+', newline ='') 
  
    # writing the min & Max Values Into The File
    with file:     
        write = csv.writer(file) 
        write.writerows(map(lambda x: [x], min_max_values)) 

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    df["Moisture 1"] = min_max_scaler.fit_transform(df[["Moisture 1"]])
    df["Moisture 2"] = min_max_scaler.fit_transform(df[["Moisture 2"]])
    df["Light"] = min_max_scaler.fit_transform(df[["Light"]])
    df["Temp"] = min_max_scaler.fit_transform(df[["Temp"]])
    df["Humidity"] = min_max_scaler.fit_transform(df[["Humidity"]])


    df.to_csv(os.path.join(outpath,"NormalizedData.csv"), index = False)

    return df
