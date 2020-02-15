from DataSet import DataSet
from TwoBayes import TwoBayes
import pandas as pd
from Metrics import Metrics


def main():
	ds = DataSet()
	ds.loadData("adultIncomeData.csv", "sex", "income")
	ds.splitIntoTrainTest()
	tb = TwoBayes()
	tb.train(ds, ">50K.")
	tb.classify(ds, "test")
	mt = Metrics()
	print(mt.calculateAccuracy(ds))
  #print("printing tb model x\n\n\n")
  #print(tb.modelX)
 # mt.preferredTreatment(ds, tb, "two")
  #mt.preferredTreatment(ds, tb, "two")


if __name__ == "__main__":
    main()
