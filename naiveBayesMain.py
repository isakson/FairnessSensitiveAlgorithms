from DataSet import DataSet
from NaiveBayes import NaiveBayes
import pandas as pd
from Metrics import Metrics

def main():

	ds = DataSet()
	ds.loadData("ClassifiedRicciDataA.csv", "Race", "Class")
	nb = NaiveBayes()
	nb.train(ds, nb.model)

	nb.classify(ds)
	mt = Metrics()
	mt.calculateAccuracy(ds)


if __name__== "__main__":
	main()