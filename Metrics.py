import pandas as pd
from DataSet import DataSet
from scipy import stats

class Metrics:

	#TODO: Maybe make the DataSet be an instance variable?

	def __init__(self):
		pass

	'''
	Calculates basic accuracy for a classified DataSet. Basic accuracy is defined as the number of classifications 
	that match the known ground truth values.
		dataSet (DataSet) - the classified DataSet to calculate accuracy for
	Note: This function assumes that the column header for Bayes classifications is "Bayes Classification"
	'''
	def calculateAccuracy(self, dataSet):
		numCorrect = 0
		dataFrame = dataSet.dataFrame

		for i in range(dataFrame.shape[0]):
			groundTruth = dataFrame.at[i, dataSet.trueLabels]
			bayesClassification = dataFrame.at[i, "Bayes Classification"]

			if groundTruth == bayesClassification:
				numCorrect += 1

		accuracy = numCorrect / dataFrame.shape[0] * 100
		print("Percentage accuracy: " + str(accuracy) + "%")
		return accuracy

	'''
	Calculates the true positive or negative rate for a classified DataSet. 
		dataSet (DataSet) - the classified DataSet to calculate true positive or negative rate for
		truePosOrNeg (int or string) - the true positive value or true negative value
	Note: this function currently only supports classifications with two possible outcomes (e.g. 0 or 1).
	This function also assumes that the column header for Bayes classifications is "bayesClassification"
	'''
	def truePosOrNeg(self, dataSet, truePosOrNeg):
		dataFrame = dataSet.dataFrame
		possibleClassifications = dataFrame["Bayes Classification"].unique()
		if len(possibleClassifications) != 2:
			return "Cannot calculate true positive or negative rate for nonbinary classifications"
		else:
			matchesLabel = 0
			actualLabel = 0 #actual number of people with this classification in the DataSet

			for i in range(dataFrame.shape[0]):
				groundTruth = dataFrame.at[i, dataSet.trueLabels]
				bayesClassification = dataFrame.at[i, "Bayes Classification"]

				if bayesClassification == truePosOrNeg and groundTruth == truePosOrNeg:
					matchesLabel +=1
				if groundTruth == truePosOrNeg:
					actualLabel +=1

			return matchesLabel, actualLabel


	'''
	Takes the information from truePosOrNeg and turns it into a rate
	'''
	def truePosOrNegRate(self, matchesLabel, actualLabel):
		return matchesLabel / actualLabel * 100

	'''
	Segments the overarching DataSet into smaller (non-overlapping) subsets by protected attribute. Returns a list of these
	subsets as well as a list of the corresponding protected attribute values.
		dataSet (DataSet) - the overarching DataSet NOTE WE'RE PROBABLY GETTING RID OF THIS PART WHEN WE MAKE IT AN INSTANCE VARIABLE
	'''
	def determineGroups(self, dataSet):
		df = dataSet.dataFrame
		possibleGroups = df[dataSet.protectedAttributes[0]].unique()

		organizedDataSetList = []
		for value in possibleGroups:
			# Setting up the group as a new DataSet
			newDataSet = DataSet()
			newDataSet.fileName = dataSet.fileName
			newDataSet.dataFrame = df[df[dataSet.protectedAttributes[0]] == value]
			newDataSet.protectedAttributes = dataSet.protectedAttributes
			newDataSet.trueLabels = dataSet.trueLabels
			newDataSet.headers = dataSet.headers
			newDataSet.numAttributes = dataSet.numAttributes

			#resets indices for later indexing
			newDataSet.dataFrame.reset_index(inplace=True, drop=True)

			organizedDataSetList.append(newDataSet)

		return organizedDataSetList, possibleGroups


	#TODO: write this comment

	def chiSquare(self, truePosByAttribute, trueNegByAttribute, TPTotal, TNTotal):
		keys = truePosByAttribute.keys()
		totalDict = {}
		for item in keys:
			posVal = truePosByAttribute[item]
			negVal = trueNegByAttribute[item]
			totalDict[item] = posVal + negVal

		overallTotal = sum(totalDict.values())

		chiSquare = 0
		for item in keys:
			expectedPosValue = (totalDict[item] * TPTotal) / overallTotal
			posOutcome = ((truePosByAttribute[item] - expectedPosValue) ** 2) / truePosByAttribute[item]

			expectedNegValue = (totalDict[item] * TNTotal) / overallTotal
			negOutcome = ((trueNegByAttribute[item] - expectedNegValue) ** 2) / trueNegByAttribute[item]

			chiSquare += posOutcome + negOutcome

		return chiSquare

	'''
	Equality of Opportunity is the metric that stipulates that the true positive rate for each protected attribute should
	be the same or similar within reason (NOTE WE WILL FIGURE THAT SHIT OUT LATER). Returns a dictionary of the true positive
	rate for each protected attribute value (where the protected attribute value is the key and the true positive rate is the value).
		dataSet (DataSet) - the overarching DataSet NOTE WE'RE PROBABLY GETTING RID OF THIS PART WHEN WE MAKE IT AN INSTANCE VARIABLE
	'''
	#TODO: test this lol
	def runEquOfOpportunity(self, dataset):
		groupDataSets, possibleGroups = self.determineGroups(dataset)
		truePosByAttribute = {}
		TPTotal = 0
		for i in range(len(groupDataSets)):
			#TODO: for now we are hardcoding in a 1 since Equality of Opportunity requests true positive rate
			truePosCount = self.truePosOrNeg(groupDataSets[i], 1)[0]
			truePosByAttribute[possibleGroups[i]] = truePosCount
			TPTotal += truePosCount

		TNTotal = 0
		trueNegByAttribute = {}
		for i in range(len(groupDataSets)):
			# TODO: for now we are hardcoding in a 0 since Equality of Opportunity requests true positive rate
			trueNegCount = self.truePosOrNeg(groupDataSets[i], 0)[0]
			trueNegByAttribute[possibleGroups[i]] = trueNegCount
			TNTotal += trueNegCount

		chiSquareVal = self.chiSquare(truePosByAttribute, trueNegByAttribute, TPTotal, TNTotal)

		#to find degreeOfFreedom, multiply rows - 1 by columns - 1, num rows is always 2, thus this is just columns - 1
		degreeOfFreedom = len(keys) - 1

		#finds p value using the cumulative distribution function
		pValue = 1 - stats.chi2.cdf(chiSquareVal, degreeOfFreedom)

		EquOfOpp = False
		if pValue < 0.05:
			EquOfOpp = True

		return pValue, EquOfOpp


