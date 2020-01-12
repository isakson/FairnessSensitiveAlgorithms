import pandas as pd
import matplotlib.pyplot as plt

## Our assumption for the moment is that the array containing the classification made by the classifier
## will be added onto the local dataframe in our dataset (to remove at the end, if needed)

class DImath:

    def __init__(self, dataSet):
        self.ds = dataSet
        self.df = dataSet.dataFrame

    def calculateConditionalProb(self, attribute, attributeValue, outcomeVal):
        numerator = (self.df.loc[(self.df[attribute]==attributeValue) &
                                 (self.df[self.ds.trueLabels] == outcomeVal), attribute]).count()
        denominator = self.df.loc[self.df[attribute]==attributeValue, attribute].count()
        return numerator/denominator

    def calculateBER(self, classifications, attribute):
        numerator = 0
        denominator = 0
        for classification in self.df[classifications].unique():
            for value in self.df[attribute].unique():
                if (classification != value):
                    numerator += self.calculateConditionalProb(attribute, value, classification)
                    denominator += 1
        if denominator != 0:
            return numerator / denominator
        return "ERROR: denominator is 0"







