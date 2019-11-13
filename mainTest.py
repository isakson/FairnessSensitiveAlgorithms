from DataSet import DataSet
from NaiveBayes import NaiveBayes
import pandas as pd

def main():

	ds = DataSet()
	ds.loadData("RicciData.csv", ["Race"], ["Position"])
	nb = NaiveBayes()
	nb.train(ds)

	ds.stripOfGroundTruth()
	nb.classify(ds)


if __name__== "__main__":
	main()