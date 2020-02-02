from DataSet import DataSet
from RepairData import RepairData
from NaiveBayes import NaiveBayes
from Metrics import Metrics
from classifierForDI import detectDI


#TODO: Finish comment
#TODO: add modified Bayes
#TODO: add "run all metrics" function
'''
Parameters:
    feldman (bool)
    bayes (naive, modified, etc.)
run all metrics all the time?
#NOTE: may overwrite previous files
'''
def pipeline(fileName, nameForFiles, protectedAttribute, trueLabels, feldman, bayes):
    # Load data into DataSet
    ds = DataSet()
    ds.loadData(fileName, protectedAttribute, trueLabels)

    DIresult = detectDI(ds)
    print(DIresult)

    # Feldman repair algorithm
    currDataSet = ds
    if feldman:
        repair = RepairData()
        repair.runRepair(ds.fileName, ds.protectedAttribute, ds.trueLabels, noiseScale=.01)
        # Pickle the Feldman-repaired data
        repair.dataSetCopy.savePickle(str(nameForFiles + "_feldman"))
        repair.dataSetCopy.saveToCsv(str(nameForFiles + "_feldman"))
        currDataSet = repair.dataSetCopy

    if bayes == "naive":
        nb = NaiveBayes()
        nb.train(currDataSet)
        nb.classify(currDataSet)

    elif bayes == "modified":
        pass


    currDataSet.savePickle(str(nameForFiles + "_" + bayes))
    currDataSet.saveToCsv(str(nameForFiles + "_" + bayes))

    # Metrics
    metrics = Metrics()
    # NOTE: this function just prints the accuracy, change to run all metrics later
    metrics.calculateAccuracy(currDataSet)


# pipeline("income-subset.csv", "filename", "sex", "income", True, "naive")