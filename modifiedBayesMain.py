from DataSet import DataSet
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
import pandas as pd
from Metrics import Metrics


'''Modified Bayes main takes in a dataset and calls the modifying functions '''
def main():

	ds = DataSet()
	mb = ModifiedBayes(ds, "income-bracket-data.csv", "sex", "income")
	mb.modify(ds, ">50k")
	mt = Metrics()
	mt.calculateAccuracy(ds)


if __name__== "__main__":
	main()
