from DataSet import DataSet
from RepairData import RepairData
from NaiveBayes import NaiveBayes
from ModifiedBayes import ModifiedBayes
from TwoBayes import TwoBayes
from Metrics import Metrics
from classifierForDI import detectDI


#TODO: add modified Bayes
#TODO: add "run all metrics" function
'''
Parameters:
    fileName (string) - The path to the file whose data should be loaded
    nameForFiles (string) - The name to assign to files written by the pipeline
    protectedAttribute (string) - The header name for the column containing the 
        protectedAttribute data in the dataset
    trueLabels (string) - The header name for the column containing the true classifications
        in the dataset
    feldman (bool) - If True, run Feldman repair algorithm. If False, do not run Feldman repair algorithm.
    bayes (string) - If "naive", run Naive Bayes. If "modified", run Modified Bayes. If "two", run Two Bayes
Notes: 
    Results (e.g. DI detector, metrics results) will be written to the results/ directory
    Pickled objects will be written to the pickledObjects/ directory
    CSVs of data will be written to the dataCSVs/ directory
'''
def pipeline(fileName, nameForFiles, protectedAttribute, trueLabels, feldman, bayes):
    # Load data into DataSet
    ds = DataSet()
    ds.loadData(fileName, protectedAttribute, trueLabels)

    # Open a file for writing results
    f = open("results/" + nameForFiles + ".txt", "w")

    DIresult = detectDI(ds)
    f.write("DI results: " + DIresult)

    # Feldman repair algorithm
    currDataSet = ds
    if feldman:
        repair = RepairData()
        repair.runRepair(ds.fileName, ds.protectedAttribute, ds.trueLabels, noiseScale=.01)
        # Pickle the Feldman-repaired data
        repair.dataSetCopy.savePickle("pickledObjects/repairedData/" + nameForFiles)
        repair.dataSetCopy.saveToCsv("dataCSVs/repairedData/" + nameForFiles + ".csv")
        currDataSet = repair.dataSetCopy

    if bayes == "naive":
        bayesObject = NaiveBayes()
        bayesObject.train(currDataSet)
        bayesObject.classify(currDataSet)

    elif bayes == "modified":
        bayesObject = ModifiedBayes()
        bayesObject.train(currDataSet, 1)

    else:
        bayesObject = TwoBayes()
        bayesObject.train(currDataSet, 1)


    currDataSet.savePickle("pickledObjects/classifiedData/" + nameForFiles)
    currDataSet.saveToCsv("dataCSVs/classifiedData/" + nameForFiles + ".csv")

    # Metrics
    metrics = Metrics()
    # NOTE: this function just prints the accuracy, change to run all metrics later
    metrics.calculateAccuracy(currDataSet)

    f.close()