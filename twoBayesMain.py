from DataSet import DataSet
from TwoBayes import TwoBayes
import pandas as pd
from Metrics import Metrics

def main():

	ds = DataSet()
	ds.loadData("income-subset.csv", "sex", "income")
	tb = TwoBayes()
	tb.train(ds, ">50K.")

	mt = Metrics()
#	mt.calculateAccuracy(ds)


if __name__== "__main__":
	main()