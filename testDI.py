from DataSet import DataSet

def main():
    ds = DataSet()
    ds.loadData("ClassifiedRicciData.csv", ["Race"], "Class")

    copy = ds.copyDataSet()
    print("protectedAttr ", copy.protectedAttributes)
    copy.makeNumerical(copy.protectedAttributes[0])
    dummified = copy.dummify()
    print(dummified.to_string())
    # print(copy.dataFrame)



    # math = DImath(ds)
    # #male0s = math.calculateConditionalProb("Gender", "M", 0)
    # ber = math.calculateBER("Classifier", "Gender")
    # print(ber)


if __name__ == "__main__":
    main()
