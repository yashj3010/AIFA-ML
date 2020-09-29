import pandas as pd
import TimeStampConverter as tsc
import os

Restfile_path=r"CSVs\source.xlsx"   #path of csvs
output_path=r"CSVs"

def meanobservation(inpath, outpath):
    df = tsc.TimeStampFormatter(inpath)
    res_df = df.groupby(pd.Grouper(key="TimeStamp")).mean()
    res_df.to_csv(os.path.join(outpath,"res_df.csv"))


if __name__ == "__main__":
    meanobservation(file_path, output_path)