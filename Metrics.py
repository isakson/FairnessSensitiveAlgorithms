import pandas as pd
from DataSet import DataSet

class Metrics:

	#TODO: Maybe make the DataSet be an instance variable?

	def __init__(self):
		pass

	'''
	Calculates basic accuracy for a classified DataSet. Basic accuracy is defined as the number of classifications 
	that match the known ground truth values.
		dataSet (DataSet) - the classified DataSet to calculate accuracy for
	Note: This function assumes that the column header for Bayes classifications is "bayesClassification"
	'''
	def calculateAccuracy(self, dataSet):
		numCorrect = 0
		dataFrame = dataSet.dataFrame

		for i in range(dataFrame.shape[0]):
			groundTruth = dataFrame.at[i, dataSet.trueLabels]
			bayesClassification = dataFrame.at[i, "bayesClassification"]

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
		possibleClassifications = dataFrame["bayesClassification"].unique()
		if len(possibleClassifications) != 2:
			return "Cannot calculate true positive or negative rate for nonbinary classifications"
		else:
			bothPosOrNeg = 0
			realPosOrNeg = 0 #actual number of people with this classification in the DataSet

			for i in range(dataFrame.shape[0]):
				groundTruth = dataFrame.at[i, dataSet.trueLabels]
				bayesClassification = dataFrame.at[i, "bayesClassification"]

				if bayesClassification == truePosOrNeg and groundTruth == truePosOrNeg:
					bothPosOrNeg +=1
				if groundTruth == truePosOrNeg:
					realPosOrNeg +=1

			return bothPosOrNeg/realPosOrNeg * 100

	'''
	Segments the overarching DataSet into smaller (non-overlapping) subsets by protected attribute. Returns a list of these
	subsets as well as a list of the corresponding protected attribute values.
		dataSet (DataSet) - the overarching DataSet NOTE WE'RE PROBABLY GETTING RID OF THIS PART WHEN WE MAKE IT AN INSTANCE VARIABLE
	'''
	def determineGroups(self, dataSet):
		df = dataSet.dataFrame
		possibleGroups = df[dataSet.protectedAttributes].unique()

		organizedDataSetList = []
		for value in possibleGroups:
			# Setting up the group as a new DataSet
			newDataSet = DataSet()
			newDataSet.fileName = dataSet.fileName
			newDataSet.dataFrame = df[df.protectedAttributes[0] == value]
			newDataSet.protectedAttributes = dataSet.protectedAttributes
			newDataSet.trueLabels = dataSet.trueLabels
			newDataSet.headers = dataSet.headers
			newDataSet.numAttributes = dataSet.numAttributes

			organizedDataSetList.append(newDataSet)

		return organizedDataSetList, possibleGroups

	'''
	Equality of Opportunity is the metric that stipulates that the true positive rate for each protected attribute should
	be the same or similar within reason (NOTE WE WILL FIGURE THAT SHIT OUT LATER). Returns a dictionary of the true positive
	rate for each protected attribute value (where the protected attribute value is the key and the true positive rate is the value).
		dataSet (DataSet) - the overarching DataSet NOTE WE'RE PROBABLY GETTING RID OF THIS PART WHEN WE MAKE IT AN INSTANCE VARIABLE
	'''
	def runEquOfOpportunity(self, dataset):
		groupDataSets, possibleGroups = self.determineGroups(dataset)
		truePosByAttribute = {}
		for i in range(len(groupDataSets)):
			#TODO: for now we are hardcoding in a 1 since Equality of Opportunity requests true positive rate
			truePosRate = self.truePosOrNeg(groupDataSets[i], 1)
			truePosByAttribute[possibleGroups[i]] = truePosRate

		return truePosByAttribute


