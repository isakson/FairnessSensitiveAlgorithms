import pandas as pd
import numpy as np
import pickle

'''
An object representing a data set pulled from a csv file. It contains:
    fileName (string) - the name of the csv file containing the data
    dataFrame (pandas DataFrame) - the DataFrame containing the data
    protectedAttribute (string) - the name of the column containing protected attribute
    trueLabels (string) - the name of the column containing the ground truth; may be None if data has been
        stripped of ground truth
    headers (array of strings) - the names of all of the columns
    numAttributes (int) - the number of columns (or attributes) in the DataFrame
'''
class DataSet:
    def __init__(self):
        pass

    '''
    Loads data into dataFrame of DataSet; sets all object instance variables
        fileName (string) - the name of the file to import
        protectedAttribute (string) - the name of the column header for the protected
            attribute
        trueLabels (string) - the column header for the ground truth in the data
    '''
    def loadData(self, fileName, protectedAttribute, trueLabels):
        self.fileName = fileName
        self.dataFrame = pd.read_csv(fileName, sep=",")
        self.protectedAttribute = protectedAttribute
        self.trueLabels = trueLabels
        self.headers = list(self.dataFrame.columns.values)
        self.numAttributes = len(self.headers)

    '''
    Adds random noise to a column in the DataFrame according to the provided scale
        columnName (string) - the name of the column where we should add noise
        scale (float) - the standard deviation (spread or “width”) of the distribution
    '''
    def addRandomNoise(self, columnName, scale):
        noise = np.random.normal(0, scale, self.dataFrame.shape[0])
        for i in range(len(noise)):
            self.dataFrame.loc[[i], [columnName]] += noise[i]

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
        newDataSet.protectedAttribute = self.protectedAttribute
        newDataSet.trueLabels = self.trueLabels
        newDataSet.headers = self.headers
        newDataSet.numAttributes = self.numAttributes
        return newDataSet

    '''
    Returns a list of all column headers with strictly numerical data.
    Note: the column containing ground truth values is not returned in this list
    '''
    def getNumericalColumns(self):
        numericalColumns = []
        for i in range(len(self.headers)):
            dataType = self.dataFrame[self.headers[i]].dtype
            column = self.headers[i]
            if column != self.trueLabels and (dataType == 'float64' or dataType == 'int64'):
                numericalColumns.append(column)
        return numericalColumns

    '''
    Returns True if a column has numerical data; returns False otherwise
        column (string) - the header for the column in question
    '''
    def isNumerical(self, column):
        dataType = self.dataFrame[column].dtype
        if dataType == 'float64' or dataType == 'int64':
            return True
        else:
            return False

    ''''
    Convert the values of a categorical variable to numbers, starting from 0. 
    This function does NOT dummify the variable. It performs the conversion in place
    (the DataSet's dataFrame object is modified directly; the function returns nothing)
        column (string) - the header for the column to convert
    '''
    def makeNumerical(self, column):
        uniqueValues = self.dataFrame[column].unique()
        numericalValues = []
        for i in range(len(uniqueValues)):
            numericalValues.append(i)
        self.dataFrame[column].replace(uniqueValues, numericalValues, inplace=True)

    '''
    Dummifies all non-numerical columns in the DataSet object's DataFrame EXCEPT the protected attribute 
    column.
    Returns the modified DataFrame object
    '''
    def dummify(self):
        columns = []
        for column in self.headers:
            if not (self.isNumerical(column) or column == self.protectedAttributes[0]):
                columns.append(column)
        return pd.get_dummies(self.dataFrame, columns=columns)

    '''
    Saves the dataFrame as a .csv file. All objects will be saved to the folder dataCSVs.
        fileName (string) - the desired output file name (note: should end in .csv, otherwise it saves as a textfile but is still comma-separated)
    '''
    def saveToCsv(self, fileName="dataFrame.csv"):
        path = "dataCSVs/" + fileName
        self.dataFrame.to_csv(path)

    '''
    Saves the current DataSet object as a pickle. All objects will be saved to the folder pickledObjects.
        fileName (string) - The DataSet object to pickle
    '''
    def savePickle(self, fileName="pickledDataSet"):
        path = "pickledObjects/" + fileName
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()
