import pandas as pd
from DataSet import DataSet
from scipy import stats

class Metrics:

	# TODO: Maybe make the DataSet be an instance variable?
	# TODO: Write function to run all metrics

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

	'''
	Performs a chi square test.
		truePosByAttribute (dict) - dictionary containing keys of protected attribute values and values of true 
			positive counts for that particular protected attribute value
		trueNegByAttribute (dict) - dictionary containing keys of protected attribute values and values of true 
			negative counts for that particular protected attribute value
		TPTotal (int) - total number of true positives
		TNTotal (int) - total number of true negatives
	
	return: chi square value (float)
	'''
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
	be the same or similar within reason.
		dataSet (DataSet) - the overarching DataSet NOTE WE'RE PROBABLY GETTING RID OF THIS PART WHEN WE MAKE IT AN INSTANCE VARIABLE
		
	return: the p value (float) and whether or not the p value implies that the statistic is significant (bool).
	'''
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

		keys = truePosByAttribute.keys()
		chiSquareVal = self.chiSquare(truePosByAttribute, trueNegByAttribute, TPTotal, TNTotal)

		# to find degreeOfFreedom, multiply rows - 1 by columns - 1, num rows is always 2, thus this is just columns - 1
		degreeOfFreedom = len(keys) - 1

		# finds p value using the cumulative distribution function
		pValue = 1 - stats.chi2.cdf(chiSquareVal, degreeOfFreedom)

		EquOfOpp = False
		if pValue < 0.05:
			EquOfOpp = True

		return pValue, EquOfOpp

	# TODO: train vs test part of dataset? help?
	'''
	Computes the fairness of trainedBayes according to counterfactual measures by running Bayes on the original data, then
	swapping the protected attribute values and reclassifying (without retraining). Then, to compute accuracy, it treats the
	original classifications as if they were true.
		dataSet (DataSet) - the original dataset
		trainedBayes (Bayes object) - the trained Bayes model
		
	return: accuracy
	'''
	def counterfactualMeasures(self, dataSet, trainedBayes):
		swappedDataSet = self.swapProtectedAttributes(dataSet)

		# TODO: Run trainedBayes on dataSetCopy
		# TODO: Figure out how the cross-validation part will work with which part of the dataset we use (since we're not retraining)

	# TODO: dealing with nonbinary protected attributes
	'''
	Copies the dataset, then swaps the protected attribute values in the copy.
		dataSet (DataSet) - the original dataset
		
	returns: a copy of dataSet with the protected attribute values swapped.
	'''
	def swapProtectedAttributes(self, dataSet):
		# TODO: change the protectedAttributes parts later
		dataSetCopy = dataSet.copyDataSet()
		dataFrame = dataSetCopy.dataFrame

		possibleAttributeValues = dataFrame[dataSetCopy.protectedAttributes[0]].unique()

		if len(possibleAttributeValues) != 2:
			return "Cannot calculate counterfactual measures for nonbinary protected attributes."

		for i in range(dataFrame.shape[0]):
			protectedAttributeValue = dataFrame.at[i, dataSetCopy.protectedAttributes[0]]
			if protectedAttributeValue == possibleAttributeValues[0]:
				dataFrame.loc[[i], [dataSetCopy.protectedAttributes[0]]] = possibleAttributeValues[1]

			else:
				dataFrame.loc[[i], [dataSetCopy.protectedAttributes[0]]] = possibleAttributeValues[0]

		return dataSetCopy

	# TODO: make this nonbinary; currently only works for binary protected attributes :/
	'''
	Counts the total number of preferred outcomes for a particular protected attribute class 
		dataset (DataSet) - the original dataset
		
	returns: a dictionary where the keys are protected attribute values and the values are the number of positive
		outcomes that protected attribute group received.
	'''
	def countPositiveOutcomes(self, dataSet):
		dataFrame = dataSet.dataFrame
		possibleAttributes = dataFrame[dataSet.protectedAttributes[0]].unique()

		posOutcomes0 = 0
		posOutcomes1 = 0
		for i in range(dataFrame.shape[0]):
			bayesClassification = dataFrame.at[i, "Bayes Classification"]
			protectedAttribute = dataFrame.at[i, dataSet.protectedAttributes[0]]

			if bayesClassification == 1 and protectedAttribute == possibleAttributes[0]:
				posOutcomes0 += 1
			elif bayesClassification == 1 and protectedAttribute == possibleAttributes[1]:
				posOutcomes1 += 1

		posOutcomesDict = {possibleAttributes[0]: posOutcomes0, possibleAttributes[1]: posOutcomes1}

		return posOutcomesDict

	# TODO: Finish this function
	'''
	Calculates whether or not a particular classification algorithm gives preferred treatment for a particular group
		dataSet (DataSet) - the original dataset
		trainedBayes (trained Bayes model) - the trained Bayes model
		typeOfBayes (string) - the name of the type of Bayes algorithm used ("naive", "modified", or "two")
		
	returns: a boolean
	'''
	def preferredTreatment(self, dataSet, trainedBayes, typeOfBayes):
		if typeOfBayes != "two":
			return True

		else:
			# Count the amount of positive outcomes each protected attribute value group receives in the original dataset
			originalPosOutcomes = self.countPositiveOutcomes(dataSet)
			dataSetCopy = dataSet.copyDataSet()
			# Change which Bayes is being run on a particular protected attribute

			# Count the amount of positive outcomes each protected attribute value group receives in the new dataset
			swappedPosOutcomes = self.countPositiveOutcomes(dataSetCopy)
			# Compare that to the original
			# If swapped is worse than the original for all, return true; else return false
			keys = originalPosOutcomes.keys()
			listOfBools = []
			for key in keys:
				if originalPosOutcomes[key] >= swappedPosOutcomes[key]:
					listOfBools.append(True)
				else:
					listOfBools.append(False)

			return all(listOfBools)




