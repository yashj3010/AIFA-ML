import pandas as pd
import TimeStampConverter as tsc
import os

file_path=r"C:\\Users\\prach\\Downloads\\Logger 1"   #path of csvs
output_path=r"C:\\Users\\prach\\Downloads"

def meanobservation(inpath, outpath):
    df = tsc.TimeStampFormatter(inpath)
    res_df = df.groupby(pd.Grouper(key="TimeStamp")).mean()
    res_df.to_csv(os.path.join(outpath,"res_df.csv"))


if __name__ == "__main__":
    meanobservation(file_path, output_path)