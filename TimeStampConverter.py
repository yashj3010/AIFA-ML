import pandas as pd
import os


file_path=r"C:\\Users\\prach\\Downloads\\Logger 1"   #path of csvs


### Fuction to convert timestamp from object datatype to datetime


def TimeStampFormatter(path):
    raw_csvs = [f for f in os.listdir(path)]
    for i in raw_csvs:
        df= pd.read_csv(os.path.join(path,i))
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%H:%M")   #Edit format according to the need currently output in HH:MM form
        df['TimeStamp'] = pd.DatetimeIndex(df['TimeStamp']).time
        return df

if __name__ == "__main__":
    TimeStampFormatter(file_path)


