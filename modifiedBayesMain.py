from DataSet import DataSet
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
import pandas as pd


'''Modified Bayes main takes in a dataset and calls the modifying functions '''
def main():

	ds = DataSet()
	mb = ModifiedBayes(ds, "income-bracket-data.csv", "sex", "income")
	mb.modify(ds, ">50k")


if __name__== "__main__":
	main()
