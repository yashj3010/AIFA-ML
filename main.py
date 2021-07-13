import DataReduction as dr
import TimeStampConverter as tsc
import DataNormalization as dn
import TargetOutcome as to
import os
import Shuffle as sh


RawFilePath = r"CSVs\Source\RawDataSample.csv"
OutputFilePath = r"CSVs\Target"
MinVal = 537            #sensor value for maximum hydration
MaxVal = 950            #sensor value for minimum hydration


if __name__ == "__main__":

    dr.MeanObservation(RawFilePath,OutputFilePath)
    ReducedFilePath = os.path.join(OutputFilePath, "ReducedData.csv")


    tsc.TimeStampFormatter(ReducedFilePath, OutputFilePath)
    FormattedFilePath = os.path.join(OutputFilePath, "TimeStampFormatted.csv")


    dn.NormalizingValues(FormattedFilePath, OutputFilePath, MinVal, MaxVal)         # First scales moisture values between given range, then normalizes data
    NormalizedFilePath = os.path.join(OutputFilePath, "NormalizedData.csv")


    to.DecisionMaker(NormalizedFilePath, OutputFilePath)
    OutcomeGeneratedFilePath = os.path.join(OutputFilePath, "TargetOutcomeGenerated.csv")


    sh.NNInputFile(OutcomeGeneratedFilePath, OutputFilePath)