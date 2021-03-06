from DataSet import DataSet
import numpy as np
from scipy import stats
from scipy import spatial
from scipy.stats import zscore
from matplotlib import pyplot

class Metrics:

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
		dataFrame = dataSet.testDataFrame

		for i in range(dataFrame.shape[0]):
			groundTruth = dataFrame.at[i, dataSet.trueLabels]
			bayesClassification = dataFrame.at[i, "Bayes Classification"]

			if groundTruth == bayesClassification:
				numCorrect += 1

		accuracy = numCorrect / dataFrame.shape[0] * 100
		return accuracy

	'''
	Calculates the true positive or negative rate for a classified DataSet.
		dataSet (DataSet) - the classified DataSet to calculate true positive or negative rate for
		truePosOrNeg (int or string) - the true positive value or true negative value in the dataset
	Returns: the number of classifications that match the truePosOrNeg and the actual number of people who have 
	that classification in the dataset.
	Note: this function currently only supports classifications with two possible outcomes (e.g. 0 or 1).
	This function also assumes that the column header for Bayes classifications is "Bayes Classification"
	'''
	def truePosOrNeg(self, dataSet, truePosOrNeg):
		dataFrame = dataSet.testDataFrame
		possibleClassifications = dataFrame["Bayes Classification"].unique()
		if len(possibleClassifications) > 2:
			return "Cannot calculate true positive or negative rate for nonbinary classifications"
		else:
			matchesLabel = 0
			actualLabel = 0  # actual number of people with this classification in the DataSet

			for i in range(dataFrame.shape[0]):
				groundTruth = dataFrame.at[i, dataSet.trueLabels]
				bayesClassification = dataFrame.at[i, "Bayes Classification"]

				if bayesClassification == truePosOrNeg and groundTruth == truePosOrNeg:
					matchesLabel +=1
				if groundTruth == truePosOrNeg:
					actualLabel +=1
			return matchesLabel, actualLabel


	'''
	Takes the information from truePosOrNeg and turns it into a rate.
		matchesLabel (int) - the number of classifications that match the truePosOrNeg
		actualLabel (int) - the actual number of people who have that classification in the dataset.
	Returns: the true pos or neg as a percentage
	'''
	def truePosOrNegRate(self, matchesLabel, actualLabel):
		return matchesLabel / actualLabel * 100

	'''
	Segments the overarching DataSet into smaller (non-overlapping) DataSet subsets by protected attribute. 
		dataSet (DataSet) - the overarching DataSet
	Returns: a list of these subsets (contained in DataSet objects) and a list of the corresponding protected 
	attribute values.
	'''
	def determineGroups(self, dataSet):
		df = dataSet.testDataFrame
		possibleGroups = df[dataSet.protectedAttribute].unique()

		organizedDataSetList = []
		for value in possibleGroups:
			# Setting up the group as a new DataSet
			newDataSet = DataSet()
			newDataSet.fileName = dataSet.fileName
			newDataSet.testDataFrame = df[df[dataSet.protectedAttribute] == value]
			newDataSet.protectedAttribute = dataSet.protectedAttribute
			newDataSet.trueLabels = dataSet.trueLabels
			newDataSet.headers = dataSet.headers
			newDataSet.testHeaders = dataSet.testHeaders
			newDataSet.numAttributes = dataSet.numAttributes

			#resets indices for later indexing
			newDataSet.testDataFrame.reset_index(inplace=True, drop=True)

			organizedDataSetList.append(newDataSet)

		return organizedDataSetList, possibleGroups

	'''
	Performs a chi square test, which tests whether there is a statistically significant difference between the real and
	predicted values.
		truePosByAttribute (dict) - dictionary containing keys of protected attribute values and values of true
			positive counts for that particular protected attribute value
		trueNegByAttribute (dict) - dictionary containing keys of protected attribute values and values of true
			negative counts for that particular protected attribute value
		TPTotal (int) - total number of true positives
		TNTotal (int) - total number of true negatives
	Returns: chi square value (float)
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
			if truePosByAttribute[item] != 0:
				posOutcome = ((truePosByAttribute[item] - expectedPosValue) ** 2) / truePosByAttribute[item]
			else:
				posOutcome = 0

			expectedNegValue = (totalDict[item] * TNTotal) / overallTotal
			if trueNegByAttribute[item] != 0:
				negOutcome = ((trueNegByAttribute[item] - expectedNegValue) ** 2) / trueNegByAttribute[item]
			else:
				negOutcome = 0

			chiSquare += posOutcome + negOutcome

		return chiSquare

	'''
	Equality of Opportunity is the metric that stipulates that the true positive rate for each protected attribute should
	be the same or similar within reason. Calculates Equality of Opportunity by running a chi square test on the true 
	positive and true negative rates.
		dataset (DataSet) - the overarching DataSet
	Returns: the p value (float) and whether or not the p value implies that the statistic is significant (bool).
	'''
	def runEquOfOpportunity(self, dataset):
		groupDataSets, possibleGroups = self.determineGroups(dataset)
		truePosByAttribute = {}
		TPTotal = 0
		for i in range(len(groupDataSets)):
			truePosCount = self.truePosOrNeg(groupDataSets[i], 1)[0]
			truePosByAttribute[possibleGroups[i]] = truePosCount
			TPTotal += truePosCount

		TNTotal = 0
		trueNegByAttribute = {}
		for i in range(len(groupDataSets)):
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

	'''
	Computes the fairness of trainedBayes according to counterfactual measures by running Bayes on the original data, then
	swapping the protected attribute values and reclassifying (without retraining). Then, to compute accuracy, it treats the
	original classifications as if they were true.
		dataSet (DataSet) - the original dataset
		trainedBayes (Bayes object) - the trained Bayes model
	Returns: accuracy
	'''
	def counterfactualMeasures(self, dataSet, trainedBayes):
		swappedDataSet = self.swapProtectedAttributes(dataSet)
		swappedDF = swappedDataSet.testDataFrame
		swappedDF = swappedDF.drop(columns=["Bayes Classification", swappedDataSet.trueLabels])

		trainedBayes.classify(swappedDataSet, "test")
		swappedDF = swappedDataSet.testDataFrame

		swappedDF[swappedDataSet.trueLabels] = dataSet.testDataFrame["Bayes Classification"].astype(int)

		return self.calculateAccuracy(swappedDataSet)

	'''
	Copies the DataSet, then swaps the protected attribute values in the copy.
		dataSet (DataSet) - the original dataset
	Returns: a copy of dataSet with the protected attribute values swapped.
	'''
	def swapProtectedAttributes(self, dataSet):
		dataSetCopy = dataSet.copyDataSet()
		dataFrame = dataSetCopy.testDataFrame

		possibleAttributeValues = dataFrame[dataSetCopy.protectedAttribute].unique()

		if len(possibleAttributeValues) != 2:
			return "Cannot swap protected attributes for nonbinary protected attributes."

		for i in range(dataFrame.shape[0]):
			protectedAttributeValue = dataFrame.at[i, dataSetCopy.protectedAttribute]
			if protectedAttributeValue == possibleAttributeValues[0]:
				dataFrame.loc[[i], [dataSetCopy.protectedAttribute]] = possibleAttributeValues[1]

			else:
				dataFrame.loc[[i], [dataSetCopy.protectedAttribute]] = possibleAttributeValues[0]
		return dataSetCopy

	'''
	Counts the total number of preferred outcomes for a particular protected attribute class
		dataset (DataSet) - the original dataset
	Returns: a dictionary where the keys are protected attribute values and the values are the number of positive
		outcomes that protected attribute group received.
	'''
	def countPositiveOutcomes(self, dataSet):
		dataFrame = dataSet.testDataFrame
		possibleAttributes = dataFrame[dataSet.protectedAttribute].unique()

		# posOutcomes0 is the number of positive classifications for cases with the 0th protected attribute value
		# posOutcomes1 is the number of positive classifications for cases with the 1st protected attribute value
		posOutcomes0 = 0
		posOutcomes1 = 0
		for i in range(dataFrame.shape[0]):
			bayesClassification = dataFrame.at[i, "Bayes Classification"]
			protectedAttribute = dataFrame.at[i, dataSet.protectedAttribute]

			if bayesClassification == 1 and protectedAttribute == possibleAttributes[0]:
				posOutcomes0 += 1
			elif bayesClassification == 1 and protectedAttribute == possibleAttributes[1]:
				posOutcomes1 += 1

		posOutcomesDict = {possibleAttributes[0]: posOutcomes0, possibleAttributes[1]: posOutcomes1}

		return posOutcomesDict

	'''
	Calculates whether or not a particular classification algorithm gives preferred treatment for a particular group
		dataSet (DataSet) - the original dataset
		trainedBayes (Bayes object) - the bayes object
		privilegedValue (String) - the protected attribute value of the privileged group
		typeOfBayes (string) - the name of the type of Bayes algorithm used ("naive", "modified", or "two")
	Returns: a boolean representing whether or not a Bayes object has preferred treatment
	'''
	def preferredTreatment(self, dataSet, trainedBayes, typeOfBayes):
		if typeOfBayes != "two":
			return True
		else:
			# Count the amount of positive outcomes each protected attribute value group receives in the original dataset
			originalPosOutcomes = self.countPositiveOutcomes(dataSet)
			dataSetCopy = dataSet.copyDataSet()
			# Change which Bayes is being run on a particular protected attribute by swapping the models
			dataSetCopy = self.swapProtectedAttributes(dataSetCopy)
			trainedBayes.modify(dataSetCopy, 1)
			# Count the amount of positive outcomes each protected attribute value group receives in the new dataset
			swappedPosOutcomes = self.countPositiveOutcomes(dataSetCopy)
			# Since we swap the protectedAttributes, we have to swap the dictionary's keys
			swappedKeysPosOutcomes = {}
			keys = list(originalPosOutcomes.keys())
			swappedKeysPosOutcomes[keys[0]] = swappedPosOutcomes[keys[1]]
			swappedKeysPosOutcomes[keys[1]] = swappedPosOutcomes[keys[0]]
			# Compare that to the original
			# If swapped is worse than the original for all, return true; else return false
			listOfBools = []
			for key in keys:
				if originalPosOutcomes[key] >= swappedKeysPosOutcomes[key]:
					listOfBools.append(True)
				else:
					listOfBools.append(False)
			return all(listOfBools)

	'''
	Calculates the probability of a positive outcome across each protected attribute value.
		dataSet (DataSet) - the dataset
	Returns: a dictionary where the keys are the protected attribute values and the values are the positive outcome rate
	for that protected attribute value.
	'''
	def groupFairness(self, dataSet):
		dataFrame = dataSet.testDataFrame
		posOutcomes = self.countPositiveOutcomes(dataSet)
		keys = posOutcomes.keys()

		probabilitiesDict = {}
		# totalsFrame is a DataFrame where the row indices are the protected attribute values and the values in the
		# 	first column are the counts of rows from the original DataFrame with that protected attribute value
		totalsFrame = dataFrame[dataSet.protectedAttribute].value_counts()

		for key in keys:
			probabilitiesDict[key] = posOutcomes[key] / totalsFrame.loc[key]

		return probabilitiesDict

	'''
	Dummifies the categorical data and then finds the distance between each row and every other row.
		dataSet (DataSet) - the dataset.
	Returns: a distribution of all distances (list), the distances and whether or not the pair had the same outcome
	(list of tuples)
	'''
	def makeEuclideanDistribution(self, dataSet):
		df = dataSet.dummify("test", dummifyAll=True)
		dataSet.testDataFrame = df
		dataSet.resetHeaders("test")
		for header in dataSet.testHeaders:
			zscores = zscore(df[header], ddof=1)
			df.loc[:, header] = zscores

		distribution = []
		distAndOutcome = []
		for i in range(df.shape[0] - 1):
			for j in range(i + 1, df.shape[0]):
				dist = spatial.distance.seuclidean(df.loc[i], df.loc[j], [1 for col in dataSet.testHeaders])
				distribution.append(dist)
				outcome = df.at[i, "Bayes Classification"] == df.at[j, "Bayes Classification"]
				distAndOutcome.append((dist, outcome))

		return distribution, distAndOutcome

	'''
	Finds the cutoff point to determine whether or not rows are considered similar
		distribution (list) - a distribution of all distances
		quantile (float) - a value between 0 and 1 indicating the quantile
	Returns: a cutoff point (float)
	'''
	def findCutoff(self, distribution, quantile=.1):
		return np.quantile(distribution, quantile)

	'''
	Computes individual fairness metric
		dataSet (DataSet) - the dataset
	Returns: the proportion of similar pairs with the same outcome over the total number of similar pairs
	'''
	def individualFairness(self, dataSet):
		dataSetCopy = dataSet.copyDataSet()
		dataSetCopy.testDataFrame.drop(dataSetCopy.protectedAttribute, axis=1)
		distribution, distAndOutcome = self.makeEuclideanDistribution(dataSetCopy)
		cutoff = self.findCutoff(distribution)

		sameOutcome = 0
		difOutcome = 0
		for item in distAndOutcome:
			# if the difference is below the cutoff and the two rows have the same outcome
			if item[0] < cutoff and item[1]:
				sameOutcome += 1
			# if the difference is below the cutoff and the two rows have different outcomes
			elif item[0] < cutoff:
				difOutcome += 1
		return sameOutcome / (sameOutcome + difOutcome)

	'''
	Plots the distance distribution and its mean line.
		distribution (list) - The distribution to plot.
		num_bins (int; default=10) - The number of bins for the distribution
	'''
	def plotDistanceDistribution(self, distribution, num_bins=10):
		print(mean(distribution) - stdev(distribution))
		n, bins, patches = pyplot.hist(distribution, num_bins)
		pyplot.axvline(mean(distribution), color='k', linestyle='dashed', linewidth=1)
		pyplot.show()

	'''
	Runs all metrics and writes their outputs to a file.
		file (file) - an open file
		dataSet (DataSet) - a dataset
		typeOfBayes (string) - the type of Bayes being used, e.g. "naive"
		trainedBayes (trained Bayes model) - the trained Bayes model
	'''
	def runAllMetrics(self, file, dataSet, typeOfBayes, trainedBayes):
		dataSet = dataSet.copyDataSet()
		file.write("Accuracy: " + str(self.calculateAccuracy(dataSet)))
		matchesLabel, actualLabel = self.truePosOrNeg(dataSet, 1)
		file.write("True positive rate: " + str(self.truePosOrNegRate(matchesLabel, actualLabel)))
		matchesLabel, actualLabel = self.truePosOrNeg(dataSet, 0)
		file.write("True negative rate: " + str(self.truePosOrNegRate(matchesLabel, actualLabel)))
		file.write("Equality of Opportunity: " + str(self.runEquOfOpportunity(dataSet)))
		file.write("Counterfactual Measures: " + str(self.counterfactualMeasures(dataSet, trainedBayes)))
		file.write("Preferred Treatment: " + str(self.preferredTreatment(dataSet, trainedBayes, typeOfBayes)))
		file.write("Group Fairness: " + str(self.groupFairness(dataSet)))
		print("Starting individual fairness")
		file.write("Individual Fairness: " + str(self.individualFairness(dataSet)))