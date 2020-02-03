from DataSet import DataSet
from TwoBayes import TwoBayes
import pandas as pd
from Metrics import Metrics

def main():

	ds = DataSet()
	ds.loadData("ClassifiedRicciDataA.csv", "Race", "Class")
	tb = TwoBayes()
	tb.train(ds, "a")

	mt = Metrics()
#	mt.calculateAccuracy(ds)


if __name__== "__main__":
	main()