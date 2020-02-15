from DataSet import DataSet
from NaiveBayes import NaiveBayes
import pandas as pd
from Metrics import Metrics

def main():

	ds = DataSet()
	ds.loadData("adultIncomeData.csv", "sex", "income")
	ds.splitIntoTrainTest()
	nb = NaiveBayes()
	nb.train(ds, nb.model)

	nb.classify(ds, "test")
	mt = Metrics()
	print(mt.calculateAccuracy(ds))


if __name__== "__main__":
	main()