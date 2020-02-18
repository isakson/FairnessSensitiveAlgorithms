import pandas as pd
from DataSet import DataSet
from statistics import median


class RepairData:
    def __init__(self):
        pass

    '''
    Sets the instance variables for a DataSet. Also copies provided DataSet and saves it as dataSetCopy.
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
        protectedAttribute = self.dataSetOriginal.protectedAttribute
        df = self.dataSetOriginal.dataFrame

        return min(df[protectedAttribute].value_counts())

    '''
    Finds all unique attribute values in our protected attributes and then finds the distributions attached to
    those values. Also returns a list of all possible values for the current protected attribute.
        nonProtectedAttribute (string) - the name of the numerical, non-protected attribute that we want to get a distribution for
    '''
    def makeDistributions(self, nonProtectedAttribute):
        df = self.dataSetOriginal.dataFrame
        protectedAttribute = self.dataSetOriginal.protectedAttribute

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

        # A list of distributions of a protected attribute's values, organized by bucket
        bucketList = [[[] for i in range(self.maxBuckets)] for subList in bucketAssignments]

        for i in range(len(bucketAssignments)):
            for j in range(len(bucketAssignments[i])):
                # Use the bucket assignment to append the distribution value to the appropriate bucket
                bucketList[i][bucketAssignments[i][j]].append(distributions[i][j])

        minMaxList = []
        for i in range(len(bucketList)):
            distributionList = []
            for j in range(len(bucketList[i])):
                if len(bucketList[i][j]) == 0:
                    print("No items in bucket: i = " + str(i) + ", j = " +
                          str(j) + ", bucketList[i][j] = " + str(bucketList[i][j]))
                minimum = min(bucketList[i][j])
                maximum = max(bucketList[i][j])
                distributionList.append([minimum, maximum])
            minMaxList.append(distributionList)

        return bucketList, minMaxList

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
    def modifyData(self, columnName, medianDistribution, bucketList, minMaxList, attributeValues):
        df = self.dataSetCopy.dataFrame

        for i in range(df.shape[0]):
            protectedAttributeValue = df.at[i, self.dataSetCopy.protectedAttribute]
            indexForProtectedAttributeValue = attributeValues.index(protectedAttributeValue)
            currentValue = df.at[i, columnName]
            bucket = self.getBucket(currentValue, indexForProtectedAttributeValue, bucketList, minMaxList)
            df.loc[[i], [columnName]] = medianDistribution[bucket]

    '''
    Finds the index of the pre-filled bucket containing the given value
        value (float) - the value to find
        indexForProtectedAttributeValue (int) - the index within bucketList for a given protected attribute
        bucketList (list of list of list of floats) - a list of distributions of a protected
            attribute's values, organized by bucket
        minMaxList (list of list of list of floats) - a list of lists of the minimum and maximum in each bucket
    '''
    def getBucket(self, value, indexForProtectedAttributeValue, bucketList, minMaxList):
        bucketedDistribution = bucketList[indexForProtectedAttributeValue]
        minMaxSublist = minMaxList[indexForProtectedAttributeValue]

        return self.getBucketHelper(value, 0, len(bucketedDistribution) - 1, bucketedDistribution, minMaxSublist)

    '''
    Helper function for getBucket binary search.
        value (float) - the value to find
        start (int) - the index of where to start searching
        stop (int) - the index of where to stop searching
        bucketedDistribution (list of list of floats) - a distribution of a protected attribute value, organized by bucket
        minMaxSublist (list of list of floats) - a list of the minimum and maximum in each bucket
    '''

    def getBucketHelper(self, value, start, stop, bucketedDistribution, minMaxSublist):
        middleIndex = (start + stop) // 2

        if value > minMaxSublist[middleIndex][1]:
            return self.getBucketHelper(value, middleIndex + 1, stop, bucketedDistribution, minMaxSublist)

        elif value < minMaxSublist[middleIndex][0]:
            return self.getBucketHelper(value, start, middleIndex, bucketedDistribution, minMaxSublist)

        else:
            return middleIndex

    '''
    Creates a DataSet object
         fileName (string) - a file name
         protectedAttribute (string) - the name of the protected attribute
         groundTruth (string) - a 1 or 0 indicating the ground truth of a particular row
         noiseScale (float) - the standard deviation of the normal distribution used to add noise to the data
    '''
    def createDataSet(self, fileName, protectedAttribute, groundTruth, noiseScale):
        data = DataSet()
        data.loadData(fileName, protectedAttribute, groundTruth)
        numericalColumns = data.getNumericalColumns("main")
        for column in numericalColumns:
            data.addRandomNoise(column, noiseScale)
        self.setDataSetVariables(data)

    '''
    Select columns for Feldman repair
    '''
    def chooseColumnsForFeldman(self, dataSet, dataName):
        columns = dataSet.getNumericalColumns("main")

        if dataName == "Restaurant":
            # TODO: consider ZIPCODE, Latitude, Longitude, Community Board, Council District, Census Tract, BIN, BBL
            # TODO: change these columns depending on what the group decides
            return ["ZIPCODE", "Latitude", "Longitude"]
        elif dataName == "Portuguese":
            # We can repair on all numerical columns
            return columns
        elif dataName == "Credit":
            # We can repair on all numerical columns
            return columns
        elif dataName == "Income":
            # We can repair on all numerical columns
            return columns
        elif dataName == "Ricci":
            # We can repair on all numerical columns
            return columns
        elif dataName == "Jury":
            # We should not repair on "trial_id, so there are no columns to repair"
            return []
        elif dataName == "German":
            return columns
        else:
            return "Invalid dataset name."

    '''
    Repairs the data in a single column
        columnName (string) - a column header
    '''
    def repairColumn(self, columnName):
        distributions, attributeValues = self.makeDistributions(columnName)
        bucketList, minMaxList = self.bucketize(distributions)
        medianDistributions = self.findMedianDistribution(bucketList)
        self.modifyData(columnName, medianDistributions, bucketList, minMaxList, attributeValues)

    '''
    Makes DataSet object from a file, then repairs the data
         fileName (string) - a file name
         protectedAttribute (string) - the name of the protected attribute
         groundTruth (string) - a 1 or 0 indicating the ground truth of a particular row
         noiseScale (float, optional) - the standard deviation of the normal distribution used to add noise to the data
    '''
    def runRepair(self, fileName, protectedAttribute, groundTruth, dataName, noiseScale=.01):
        self.createDataSet(fileName, protectedAttribute, groundTruth, noiseScale)
        repairColumns = self.chooseColumnsForFeldman(self.dataSetCopy, dataName)
        print("Columns to repair: ", repairColumns)
        for column in repairColumns:
            self.repairColumn(column)
