from DataSet import DataSet
from NaiveBayes import NaiveBayes
import pandas as pd

def main():

	ds = DataSet()
	ds.loadData("income-subset.csv", "sex", "income")
	nb = NaiveBayes()
	nb.train(ds)

	nb.classify(ds)


if __name__== "__main__":
	main()