import pandas as pd
import math
from mpmath import *

class Bayes:

	def __init__(self):
		pass

	'''Counts the number of rows that have both a1Val and a2Val
	   a1 - name of attribute #1, a1Val - particular attribute category
	   (Ex: a1= sex, a1Val= Female '''
	def countIntersection(self, dataFrame, a1, a1Val, a2, a2Val):
		try:
			return (len(dataFrame.groupby([a1, a2]).get_group((a1Val, a2Val))))
		except:
			return 0

	'''Counts the number of rows that have aVal
	   a - name of attribute, aVal - particular attribute category
	   (Ex: a= sex, aVal= Female '''
	def countAttr(self, dataFrame, a, aVal):
		try:
			return dataFrame.loc[dataFrame[a] == aVal, a].count()
		except:
			return 0

	'''Returns the probability of a specific attribute category's probability (# +category/# people total)
	   gives a value between 0-1.
	   a - name of attribute, value - particular attribute category '''
	def attributeCategoryProbability(self, dataFrame, a, value):
		return dataFrame.loc[dataFrame[a] == value, a].count() / len(dataFrame.index)

	'''Returns an array of the unique categories (strings) in a column
	   a - name of attribute'''
	def getAttributeCategories(self, dataFrame, a):
		return dataFrame[a].unique()

	def getRares(self, dataFrame, a):
		attributeCounts = {}
		total = []
		rares = []
		for val in dataFrame[a].unique():
			attributeCounts[val] = self.countAttr(dataFrame, a, val)
			total += self.countAttr(dataFrame, a, val)
		cutoff = .01 * total
		for key in attributeCounts.keys():
			if attributeCounts[key] <= cutoff:
				rares.append(key)
		return rares

	def getRareProb(self, dataFrame, groundTruth, classification, attribute, rares):
		numerator = 0
		for val in rares:
			numerator += self.countIntersection(dataFrame, attribute, val, groundTruth, classification)
		return numerator / self.countAttr(dataFrame, groundTruth, classification)

	'''Compute the non conditional mean of attribute a '''
	def calculateMean(self, dataFrame, a):
		return dataFrame[a].mean()


	'''Compute the conditional mean of attribute a given the classification (gTValue)
	   The variance of a variable given the value of another variable'''
	def calculateConditionalMean(self, dataFrame, a, groundTruth, gTValue):
		return dataFrame.groupby([groundTruth]).get_group(gTValue)[a].mean()


	'''Compute non conditional standard deviation of attribute a'''
	def calculateStandardDeviation(self, dataFrame, a):
		stdd = dataFrame[a].std()
		return stdd

	'''Compute conditional standard deviation of attribute a given the classification'''
	def calculateConditionalStandardDeviation(self, dataFrame, a, groundTruth, gTValue):
		stdd = dataFrame.groupby([groundTruth]).get_group(gTValue)[a].std()
		return stdd



	''' P(a | b) = #rows that have both aValue and bValue / #rows with bValue '''
	def calculateCrossAttributeProbability(self, dataFrame, b, bValue, a, aValue):
		return self.countIntersection(dataFrame, a, aValue, b, bValue) / self.countAttr(dataFrame, b, bValue)

	'''Computes the gaussian probability given a particular standard deviation, mean, and a current numerical attribute value'''
	def calculateGaussianProbability(self, mean, std, value):

		zscore = ((value - mean) * (value - mean)) / ((2*(std*std)))
		gaussian = (1 / sqrt(2*pi*(std * std)) ) * (e **(-zscore))
		return gaussian
