from DataSet import DataSet
from TwoBayes import TwoBayes
import pandas as pd
from Metrics import Metrics


def main():
  ds = DataSet()
  ds.loadData("ClassifiedRicciData.csv", "Race", "Class")
  tb = TwoBayes()
  tb.train(ds, 1)
  mt = Metrics()
  mt.calculateAccuracy(ds)
  #print("printing tb model x\n\n\n")
  #print(tb.modelX)
  mt.preferredTreatment(ds, tb, "two")
  mt.preferredTreatment(ds, tb, "two")


if __name__ == "__main__":
    main()
