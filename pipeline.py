from DataSet import DataSet
from RepairData import RepairData
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
from TwoBayes import TwoBayes
from Metrics import Metrics
from classifierForDI import detectDI

'''
Parameters:
    fileName (str) - The path to the file whose data should be loaded
    nameForFiles (str) - The name to assign to files written by the pipeline
    protectedAttribute (str) - The header name for the column containing the protectedAttribute data in the dataset
    trueLabels (str) - The header name for the column containing the true classifications in the dataset
    feldman (bool) - If True, run Feldman repair algorithm. If False, do not run Feldman repair algorithm
    bayes (str) - If "naive", run Naive Bayes. If "modified", run Modified Bayes. If "two", run Two Bayes
    dataName (str) - The name for the dataset (e.g. "Restaurants" or "Jury")
Notes:
    Results (e.g. DI detector, metrics results) will be written to the results/ directory
    Pickled objects will be written to the pickledObjects/ directory
    CSVs of data will be written to the dataCSVs/ directory
'''
def pipeline(fileName, nameForFiles, protectedAttribute, trueLabels, feldman, bayes, dataName):
    # Load data into DataSet
    ds = DataSet()
    ds.loadData(fileName, protectedAttribute, trueLabels)

    # Open a file for writing results
    f = open("results/" + nameForFiles + ".txt", "w")

    print("Starting DI detection")
    DIresult = detectDI(ds)
    f.write("DI results on original data: " + DIresult)

    # Feldman repair algorithm
    currDataSet = ds
    if feldman == "yes":
        print("Starting Feldman")
        repair = RepairData()
        repair.runRepair(ds.fileName, ds.protectedAttribute, ds.trueLabels, dataName, noiseScale=.01)
        # Pickle the Feldman-repaired data
        repair.dataSetCopy.savePickle("pickledObjects/repairedData/" + nameForFiles)
        repair.dataSetCopy.saveToCsv("dataCSVs/repairedData/" + nameForFiles + ".csv")
        currDataSet = repair.dataSetCopy

        print("Starting post-Feldman DI detection")
        postFeldmanDIresult = detectDI(repair.dataSetCopy)
        f.write("DI results after Feldman: " + postFeldmanDIresult)

    #Split data into test and training set
    currDataSet.splitIntoTrainTest()
    print("Split into test train")

    #Bayes
    if bayes == "naive":
        print("Starting Naive Bayes")
        bayesObject = NaiveBayes()
        bayesObject.train(currDataSet, bayesObject.model)
        bayesObject.classify(currDataSet, "test")
        print("Completed Naive Bayes")

    elif bayes == "modified":
        bayesObject = ModifiedBayes()
        bayesObject.train(currDataSet, 1)
        bayesObject.classify(currDataSet, "test")

    else:
        bayesObject = TwoBayes()
        bayesObject.train(currDataSet, 1)
        bayesObject.classify(currDataSet, "test")

    currDataSet.savePickle("pickledObjects/classifiedData/" + nameForFiles)
    currDataSet.saveToCsv("dataCSVs/classifiedData/" + nameForFiles + ".csv")

    # Metrics
    print("Starting metrics")
    metrics = Metrics()
    metrics.runAllMetrics(f, currDataSet, bayes, bayesObject)
    print("Completed metrics")

    f.close()