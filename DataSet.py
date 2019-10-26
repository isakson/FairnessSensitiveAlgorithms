import pandas as pd

'''
An object representing a data set pulled from a csv file. It contains:
    fileName (string) - the name of the csv file containing the data
    dataFrame (pandas DataFrame) - the DataFrame containing the data
    protectedAttributes (array of strings) - the names of the columns containing protected attributes
    trueLabels (string) - the name of the column containing the ground truth; may be None if data has been
        stripped of ground truth
    headers (array of strings) - the names of all of the columns
    numAttributes (int) - the number of columns (or attributes) in the DataFrame
    hasGroundTruth (bool) - whether or not the DataFrame has a column with ground truth values
'''
class DataSet:
    def __init__(self):
        pass

    '''
    Loads data into dataFrame of DataSet; sets all object instance variables
    '''
    def loadData(self, fileName, protectedAttributes, trueLabels):
        self.fileName = fileName
        self.dataFrame = pd.read_csv(fileName, sep=",")
        self.protectedAttributes = protectedAttributes
        self.trueLabels = trueLabels
        self.headers = list(self.dataFrame.columns.values)
        self.numAttributes = len(self.headers)
        self.hasGroundTruth = True

    '''
    Creates a new DataSet that is a copy of this DataSet. 
    Returns the new DataSet
    Note: the DataSet produced will have the same fileName as the DataSet we copied from, 
    even though the data is not re-imported from the fileName
    '''
    def copyDataSet(self):
        newDataSet = DataSet()
        newDataSet.fileName = self.fileName
        newDataSet.dataFrame = self.dataFrame.copy()
        newDataSet.protectedAttributes = self.protectedAttributes
        newDataSet.trueLabels = self.trueLabels
        newDataSet.headers = self.headers
        newDataSet.numAttributes = self.numAttributes
        newDataSet.hasGroundTruth = self.hasGroundTruth
        return newDataSet

    '''
    Strips the DataSet's dataFrame of the column containing the ground truth.
    Also sets hasGroundTruth to false.
    '''
    def stripOfGroundTruth(self):
        self.dataFrame = self.dataFrame.drop(columns=[self.trueLabels])
        self.hasGroundTruth = False
        self.trueLabels = None
        self.headers = list(self.dataFrame.columns.values)
        self.numAttributes = len(self.headers)



    #TODO: create method to determine whether or not other attributes are numerical/valid or not
