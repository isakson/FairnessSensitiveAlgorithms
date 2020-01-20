import pandas as pd
import math
#Naive Bayes

class Bayes:

	def __init__(self):
		pass

	'''counts the number of rows that have both a1Val and a2Val'''
	def countIntersection(self, dataFrame, a1, a1Val, a2, a2Val):
		try:
			return (len(dataFrame.groupby([a1, a2]).get_group((a1Val, a2Val))))
		except:
			return 0

	'''counts the number of rows that have both a1Val and a2Val'''
	def countIntersectionE(self, dataFrame, a1, a1Val, a2, a2Val):
		return (len(dataFrame.groupby([a1, a2]).get_group((a1Val, a2Val))))

	'''counts the number of rows that have aVal'''
	def countAttr(self, dataFrame, a, aVal):
		try:
			return dataFrame.loc[dataFrame[a] == aVal, a].count()
		except:
			return 0

	'''returns the probability of a specific attribute category's probability (# +category/# people total)
	   gives a value between 0-1 '''
	def attributeCategoryProbability(self, dataFrame, a, value):
		#P(x2) where x2 is in a
		return dataFrame.loc[dataFrame[a] == value, a].count() / len(dataFrame.index)

	'''returns an array of the unique categories (strings) in a column '''
	def getAttributeCategories(self, dataFrame, a):
		return dataFrame[a].unique() 


	def calculateMean(self, dataFrame, a):
		return dataFrame[a].mean()


	def calculateConditionalMean(self, dataFrame, a, groundTruth, gTValue):
		return dataFrame.groupby([groundTruth]).get_group(gTValue)[a].mean()


	def calculateStandardDeviation(self, dataFrame, a):
		return dataFrame[a].std()


	def calculateConditionalStandardDeviation(self, dataFrame, a, groundTruth, gTValue):
		return dataFrame.groupby([groundTruth]).get_group(gTValue)[a].std()

	
	''' Probability of (a | b)'''
	def calculateCrossAttributeProbability(self, dataFrame, b, bValue, a, aValue):
		return self.countIntersection(dataFrame, a, aValue, b, bValue) / self.countAttr(dataFrame, b, bValue)


	def calculateGaussianProbability(self, mean, std, value):

		zscore = ((value - mean) * (value - mean)) / ((2*(std*std)))

		gaussian = (1 / math.sqrt(2*math.pi*(std * std)) ) * (math.e **(-zscore))
		
		return gaussian