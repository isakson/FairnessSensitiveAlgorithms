"""
Data Loading via pandas
10/23/19
"""
import numpy as np
import pandas as pd
import sys
from scipy.stats import percentileofscore

def main():
    fileName = sys.argv[1]
    dataFrame = pd.read_csv(fileName, sep=",")
    #dataFrameArray = dataFrame.values
    # for index, row in dataFrame.iterrows():
    #     print(row)
    #     print()

    # return dataFrame

    test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #This needs a 0 to be in the dataset??
    print(np.percentile(test, 95))
    #This needs there to not be a 0 in the dataset??
    print(percentileofscore(test, 3, kind='rank')) #weak and rank give us 30 for this example


if __name__ == "__main__":
    main()