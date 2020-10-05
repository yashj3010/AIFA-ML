import pandas as pd
import numpy as np
import os


def NNInputFile(inpath, outpath):

    df = pd.read_csv(inpath, index_col = False)

    ds = df.reindex(np.random.permutation(df.index))
    ds.to_csv(os.path.join(outpath, "ShuffledNNfile.csv"), header= False, index = False)

