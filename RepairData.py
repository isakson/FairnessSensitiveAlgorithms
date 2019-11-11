import pandas as pd
from DataSet import DataSet
from statistics import median


class RepairData:
    def __init__(self):
        pass

    '''
    Sets the instance variables for a DataSet
        dataSet (DataSet) - a DataSet object
    '''
    def setDataSetVariables(self, dataSet):

        self.dataSetOriginal = dataSet
        self.dataSetCopy = dataSet.copyDataSet()
        self.maxBuckets = self.getMaxBuckets()

    '''
    Finds the protected attribute value with the fewest individuals and returns the count of those individuals
    '''
    def getMaxBuckets(self):
        protectedAttributes = self.dataSetOriginal.protectedAttributes
        df = self.dataSetOriginal.dataFrame

        valueCounts = []
        for attribute in protectedAttributes:
            valueCounts.append(min(df[attribute].value_counts()))

        return min(valueCounts)

    '''
    Finds all unique attribute values in our protected attributes and then finds the distributions attached to
    those values. Also returns a list of all possible values for the current protected attribute.
        protectedAttribute (string) - the name of the protected attribute we want to use to make the distributions
        nonProtectedAttribute (string) - the name of the numerical, non-protected attribute that we want to get a distribution for
    '''
    def makeDistributions(self, protectedAttribute, nonProtectedAttribute):
        df = self.dataSetOriginal.dataFrame

        attributeDistributions = []
        attributeValues = []
        for value in df[protectedAttribute].unique():
            protectedDataFrame = df.loc[df[protectedAttribute] == value, [nonProtectedAttribute]]
            series = protectedDataFrame[nonProtectedAttribute].tolist()
            attributeDistributions.append(series)
            attributeValues.append(value)
        return attributeDistributions, attributeValues

    '''
    Takes the list of distributions from makeDistributions and puts the values into buckets.
        distributions (list of lists) - the values from a single column separated by a protectedAttribute value
    '''
    def bucketize(self, distributions):
        # bucketAssignments is a list containing the index values for the bucket that the distribution values should end up in
        # e.g. [0, 1, 2, 3, 0] assigns the first and last items to bucket 0, the second item to bucket 2, etc.
        bucketAssignments = []
        for i in range(len(distributions)):
            bucketAssignments.append(pd.qcut(distributions[i], self.maxBuckets, labels=False))
        print(bucketAssignments)

        # A list of distributions of a protected attribute's values, organized by bucket
        bucketList = [[[] for i in range(self.maxBuckets)] for subList in bucketAssignments]

        for i in range(len(bucketAssignments)):
            for j in range(len(bucketAssignments[i])):
                # Use the bucket assignment to append the distribution value to the appropriate bucket
                bucketList[i][bucketAssignments[i][j]].append(distributions[i][j])


        return bucketList

    '''
    Takes in bucketized values and returns a median distribution.
        bucketList (list of list of list of floats) - a list of distributions of a protected 
            attribute's values, organized by bucket
    '''
    def findMedianDistribution(self, bucketList):
        bucketMedians = [[] for subList in bucketList]
        for dist in range(len(bucketList)):
            for bucket in bucketList[dist]:
                bucketMedians[dist].append(median(bucket))
        zippedList = list(zip(*bucketMedians))

        medianDistribution = []
        for sublist in zippedList:
            medianDistribution.append(median(sublist))

        return medianDistribution

    '''
    Updates a DataSet object with modified values
        columnName (string) - a column header
        medianDistribution (list of floats) - a one-dimensional list containing the median values for each bucket
            in bucketList
        bucketList (list of list of list of floats) - a list of distributions of a protected 
            attribute's values, organized by bucket
        attributeValues (list of strings) - a list of all possible values for the current protected attribute
    '''
    def modifyData(self, columnName, medianDistribution, bucketList, attributeValues):
        df = self.dataSetCopy.dataFrame

        for i in range(df.shape[0]):
            # TODO: Note: this assumes that there is only one protected attribute
            protectedAttributeValue = df.at[i, self.dataSetCopy.protectedAttributes[0]]
            indexForProtectedAttributeValue = attributeValues.index(protectedAttributeValue)
            currentValue = df.at[i, columnName]
            bucket = self.getBucket(currentValue, indexForProtectedAttributeValue, bucketList)
            df.loc[[i], [columnName]] = medianDistribution[bucket]

    '''
    Finds the index of the pre-filled bucket containing the given value
        value (float) - the value to find
        indexForProtectedAttributeValue (int) - the index within bucketList for a given protected attribute
        bucketList (list of list of list of floats) - a list of distributions of a protected 
            attribute's values, organized by bucket
    '''
    def getBucket(self, value, indexForProtectedAttributeValue, bucketList):
        bucketedDistribution = bucketList[indexForProtectedAttributeValue]
        #TODO: Note: this will be bad for big data sets
        for bucket in bucketedDistribution:
            if value in bucket:
                return bucketedDistribution.index(bucket)

    '''
    Creates a DataSet object
         fileName (string) - a file name
         protectedAttributes (list) - a list of the names of the protected attributes 
         groundTruth (string) - a 1 or 0 indicating the ground truth of a particular row
         noiseScale (float) - the standard deviation of the normal distribution used to add noise to the data
    '''
    def createDataSet(self, fileName, protectedAttributes, groundTruth, noiseScale):
        data = DataSet()
        data.loadData(fileName, protectedAttributes, groundTruth)
        numericalColumns = data.getNumericalColumns()
        for column in numericalColumns:
            data.addRandomNoise(column, noiseScale)
        self.setDataSetVariables(data)

    '''
    Repairs the data in a single column
        columnName (string) - a column header
    '''
        #TODO: test this on its own (we didn't get to it last time :( )
    def repairColumn(self, columnName):
        distributions, attributeValues = self.makeDistributions(self.dataSetCopy.protectedAttributes, columnName)
        bucketList = self.bucketize(distributions)
        medianDistributions = self.findMedianDistribution(bucketList)
        self.modifyData(columnName, medianDistributions, bucketList, attributeValues)

    '''
    Makes DataSet object from a file, then repairs the data
         fileName (string) - a file name
         protectedAttributes (list) - a list of the names of the protected attributes 
         groundTruth (string) - a 1 or 0 indicating the ground truth of a particular row
         noiseScale (float, optional) - the standard deviation of the normal distribution used to add noise to the data
    '''
    def runRepair(self, fileName, protectedAttributes, groundTruth, noiseScale=.01):
        self.createDataSet(fileName, protectedAttributes, groundTruth, noiseScale)
        numericalColumns = self.dataSetCopy.getNumericalColumns()
        for column in numericalColumns:
            self.repairColumn(column)

    #TODO: current error: "AttributeError: 'DataFrame' object has no attribute 'unique'"

    #TODO: save repaired data as a .csv
