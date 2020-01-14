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
	This function also assumes that the column header for Bayes classifications is "Bayes Classification"
	'''
	def truePosOrNeg(self, dataSet, truePosOrNeg):
		dataFrame = dataSet.dataFrame
		possibleClassifications = dataFrame["Bayes Classification"].unique()
		if len(possibleClassifications) != 2:
			return "Cannot calculate true positive or negative rate for nonbinary classifications"
		else:
			bothPosOrNeg = 0
			realPosOrNeg = 0 #actual number of people with this classification in the DataSet

			for i in range(dataFrame.shape[0]):
				groundTruth = dataFrame.at[i, dataSet.trueLabels]
				bayesClassification = dataFrame.at[i, "Bayes Classification"]

				if bayesClassification == truePosOrNeg and groundTruth == truePosOrNeg:
					bothPosOrNeg +=1
				if groundTruth == truePosOrNeg:
					realPosOrNeg +=1

			return bothPosOrNeg/realPosOrNeg * 100