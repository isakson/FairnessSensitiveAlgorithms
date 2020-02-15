from DataSet import DataSet
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
import pandas as pd
from Metrics import Metrics


'''Create dataset object, load data to it, create modified bayes object (which creates a modified naive object) and modify it '''
def main():

	ds = DataSet()
	ds.loadData("adultIncomeData.csv", "sex", "income")
	ds.splitIntoTrainTest()
	mb = ModifiedBayes()
	mb.train(ds, ">50K.")
	mb.classify(ds, "test")
	mt = Metrics()
	print(mt.calculateAccuracy(ds))


if __name__== "__main__":
	main()
