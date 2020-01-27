from DataSet import DataSet
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
import pandas as pd
from Metrics import Metrics


'''create dataset object, load data to it, create modified bayes object (which creates a modified naive object) and modify it '''
def main():

	ds = DataSet()
	ds.loadData("income-bracket-data.csv", "sex", "income")
	mb = ModifiedBayes()
	mb.modify(ds, ">50k")
	mt = Metrics()
	mt.calculateAccuracy(ds)


if __name__== "__main__":
	main()
