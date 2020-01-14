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

	
	''' Probability of (a | groundTruth)'''
	def calculateCrossAttributeProbability(self, dataFrame, groundTruth, gTValue, a, aValue):
		#try-except in case there's no instances of the particular attribute pair
		try:
			inSameRowProb = (len(dataFrame.groupby([a, groundTruth]).get_group((aValue, gTValue)))) / len(dataFrame.index)
			aProb = self.attributeCategoryProbability(dataFrame, a, aValue)
			CAP = inSameRowProb / aProb
			return CAP
		except:
			return 0


	def calculateGaussianProbability(self, mean, std, value):

		zscore = ((value - mean) * (value - mean)) / ((2*(std*std)))

		gaussian = (1 / math.sqrt(2*math.pi*(std * std)) ) * (math.e **(-zscore))
		
		return gaussian