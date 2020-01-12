from DImath import DImath
from DataSet import DataSet

def main():
    ds = DataSet()
    ds.loadData("testRepairWithClassifications.csv", ["Gender"], "Truth")
    math = DImath(ds)
    #male0s = math.calculateConditionalProb("Gender", "M", 0)
    ber = math.calculateBER("Classifier", "Gender")
    print(ber)


if __name__ == "__main__":
    main()
