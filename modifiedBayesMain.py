from DataSet import DataSet
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
import pandas as pd


'''Modified Bayes main takes in a dataset and calls the modifying functions '''
def main():

	ds = DataSet()

	mb = ModifiedBayes(ds, "RicciDataBinary.csv", "Race", "Position")
	mb.modify(ds)


if __name__== "__main__":
	main()