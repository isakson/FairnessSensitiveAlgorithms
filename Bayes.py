from mpmath import *

class Bayes:

	def __init__(self):
		pass

	'''
	Counts the number of rows in a that have both a1Val and a2Val
		dataFrame - the DataFrame object
		a1 - column name of attribute 1
		a1Val - value of attribute 1
		a2 - column name of attribute 2
		a2Val - value of attribute 2
	(Ex: a1= sex, a1Val= Female)
	'''
	def countIntersection(self, dataFrame, a1, a1Val, a2, a2Val):
		try:
			return (len(dataFrame.groupby([a1, a2]).get_group((a1Val, a2Val))))
		except:
			return 0

	'''
	Counts the number of rows that have aVal
		dataFrame - the DataFrame object
		a - column name of attribute
		aVal - value of attribute
	(Ex: a= sex, aVal= Female)
	'''
	def countAttr(self, dataFrame, a, aVal):
		try:
			return dataFrame.loc[dataFrame[a] == aVal, a].count()
		except:
			return 0

	'''
	Returns the probability of a specific attribute value for a given category 
	(num people with given value for column/num people total).
	Returns a probability.
	   a - column name
	   value - value of attribute
	'''
	def attributeCategoryProbability(self, dataFrame, a, value):
		return dataFrame.loc[dataFrame[a] == value, a].count() / len(dataFrame.index)

	'''
	Returns an array of the unique categories (strings) in a column
		dataFrame - the DataFrame object
		a - name of attribute
	'''
	def getAttributeCategories(self, dataFrame, a):
		return dataFrame[a].unique()

	'''
	Find all "rare" values in the DataFrame, where "rare" values are any value that 
	appear in less than or equal to 1% of the rows.
	Returns a list containing all rare values. 
		dataFrame - the DataFrame object
		a - column name of attribute
	'''
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

	'''
	Find the overall probability of rare values in the DataFrame.
	Returns the probability of getting a given classification given a rare value.
		dataFrame - the DataFrame object
		groundTruth - column name of the ground truth column
		classification - a given classification (a value in the groundTruth column)
		attribute - column name containing the current rare values
		rares - a list of rare values for the column
	'''
	def getRareProb(self, dataFrame, groundTruth, classification, attribute, rares):
		numerator = 0
		for val in rares:
			numerator += self.countIntersection(dataFrame, attribute, val, groundTruth, classification)
		return numerator / self.countAttr(dataFrame, groundTruth, classification)

	'''
	Compute the mean of attribute a.
		dataFrame - the DataFrame object
		a - the column name for the attribute
	'''
	def calculateMean(self, dataFrame, a):
		return dataFrame[a].mean()

	'''
	Compute the conditional mean of attribute a given the classification (ground truth value)
	Returns: the conditional mean
		dataFrame - the DataFrame object
		a - the column name for the attribute
		groundTruth - the name of the groundTruth column
		gTValue - a given classification (a value in the groundTruth column)
	'''
	def calculateConditionalMean(self, dataFrame, a, groundTruth, gTValue):
		return dataFrame.groupby([groundTruth]).get_group(gTValue)[a].mean()

	'''
	Compute standard deviation of attribute a
	Returns: the standard deviation
		dataFrame - the DataFrame object
		a - the column name for the attribute
	'''
	def calculateStandardDeviation(self, dataFrame, a):
		stdd = dataFrame[a].std()
		return stdd

	'''
	Compute conditional standard deviation of attribute a given the classification
	Returns: the conditional standard deviation
		dataFrame - the DataFrame object
		a - the column name for the attribute
		groundTruth - the name of the groundTruth column
		gTValue - a given classification (a value in the groundTruth column)
	'''
	def calculateConditionalStandardDeviation(self, dataFrame, a, groundTruth, gTValue):
		stdd = dataFrame.groupby([groundTruth]).get_group(gTValue)[a].std()
		return stdd

	''' 
	Compute cross attribute probability --
			P(a | b) =  (number of rows that have both aValue and bValue) /  number of rows with bValue
			
	Returns: the cross attribute probability
		dataFrame - the DataFrame object
		a - the column name for attribute a
		aValue - value of attribute a
		b - the column name for attribute b
		bValue -  value of attribute b
	'''
	def calculateCrossAttributeProbability(self, dataFrame, b, bValue, a, aValue):
		return self.countIntersection(dataFrame, a, aValue, b, bValue) / self.countAttr(dataFrame, b, bValue)

	'''
	Compute Gaussian probability 
	Returns: Gaussian probability
		mean - a mean
		std - standard deviation
		value - current numerical attribute value
	'''
	def calculateGaussianProbability(self, mean, std, value):
		zscore = ((value - mean) * (value - mean)) / ((2*(std*std)))
		gaussian = (1 / sqrt(2*pi*(std * std))) * (e **(-zscore))
		return gaussian
