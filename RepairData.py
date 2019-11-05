import pandas as pd
from DataSet import DataSet


class RepairData:
    def __init__(self, dataSet):
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
    those values.
    protectedAttribute (string) - the name of the protected attribute we want to use to make the distributions
    nonProtectedAttribute (string) - the name of the numerical, non-protected attribute that we want to get a distribution for
    '''
    def makeDistributions(self, protectedAttribute, nonProtectedAttribute):
        df = self.dataSetOriginal.dataFrame

        attributeDistributions = []
        for value in df[protectedAttribute].unique():
            protectedDataFrame = df.loc[df[protectedAttribute] == value, [nonProtectedAttribute]]
            series = protectedDataFrame[nonProtectedAttribute].tolist()
            attributeDistributions.append(series)
        return attributeDistributions

    '''
    Takes the list of distributions from makeDistributions and puts the values into buckets.
    distributions (list of lists) - the values from a single column separated by a protectedAttribute value
    '''
    def bucketize(self, distributions, numBuckets=None):
        # TODO: figure out why we have empty bucket
        # TODO: remove optional parameter once we figure out what's happening
        if not numBuckets:
            numBuckets = self.maxBuckets
        # bucketAssignments is a list containing the index values for the bucket that the distribution values should end up in
        # e.g. [0, 1, 2, 3, 0] assigns the first and last items to bucket 0, the second item to bucket 2, etc.
        bucketAssignments = []
        for i in range(len(distributions)):
            bucketAssignments.append(pd.qcut(distributions[i], numBuckets, labels=False))
        print(bucketAssignments)

        bucketList = [[[] for i in range(numBuckets)] for subList in bucketAssignments]

        for i in range(len(bucketAssignments)):
            for j in range(len(bucketAssignments[i])):
                # Use the bucket assignment to append the distribution value to the appropriate bucket
                bucketList[i][bucketAssignments[i][j]].append(distributions[i][j])

        return bucketList
    #
    # def findMedianDistribution(self, bucketList):
    #     # TODO: Figure out how to do median distribution
    #     medianDistributions = []
    #     for sublist in range(len(bucketList)):
    #         for bucket in range(len(sublist)):
