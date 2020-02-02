from DataSet import DataSet
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
import pandas as pd
from Metrics import Metrics


'''Create dataset object, load data to it, create modified bayes object (which creates a modified naive object) and modify it '''
def main():

	ds = DataSet()
	ds.loadData("income-subset.csv", "sex", "income")
	mb = ModifiedBayes()
	mb.train(ds, ">50k")
	mt = Metrics()
	mt.calculateAccuracy(ds)


if __name__== "__main__":
	main()
