from Bayes import Bayes
import pandas as pd
import operator
import math
import numpy as np
import mpmath

class NaiveBayes(Bayes):

	def __init__(self):
		self.model = [] 

	'''
	Create model filled as such:
		self.model = array of attributes (e.g. race, position, etc.) where each index points to a dictionary
			attrDict (categorical) { key =  attribute category (e.g. white, black, hispanic), value = probability dictionary}
				probabilityDict = { key = classification (e.g. gets loan, differed, doesn't get loan), value = P(attrCategory|classification)}

		attrDict (numerical) { key = "mean" or "std", value = meanDict or stdDict}
			meanDict = {key = classification, value = conditional mean given this classification}
			stdDict = {key = classification, value = conditional std given this classification}
		dataSet (DataSet) - the dataset
		model (Bayesian model object) - the model to train
	'''
	def train(self, dataSet, model):
		dataFrame = dataSet.trainDataFrame
		groundTruth = dataSet.trueLabels
		classificationList = dataFrame[groundTruth].unique()

		#to ensure that we don't train twice
		if bool(model):
			print("Error: Model not empty.")
			pass

		#for each of the attributes in the datset (a1...an)
		for attribute in dataSet.trainHeaders:
			#create outermost dictionary of the model (key = attribute category, value = another dictionary)
			attrDict = {}

			#if numerical type data
			if(attribute in dataSet.getNumericalColumns("train")):
				#for each numerical attribute create dict to hold mean and standard deviation
				meanDict = {}
				stdDict = {}

				#for each of the possible classifications possible (i.e. lieutenant, captain, etc.)
				for classification in classificationList:
					#skip this case
					if(groundTruth == attribute):
						continue
					#calculate the conditional mean and standard deviation based on each classification
					mean = self.calculateConditionalMean(dataFrame, attribute, groundTruth, classification)
					std = self.calculateConditionalStandardDeviation(dataFrame, attribute, groundTruth, classification)
					meanDict[classification] = mean
					stdDict[classification] = std

				#append it to the outer dictionary
				attrDict["mean"] = meanDict
				attrDict["std"] = stdDict

			#categorical type data
			else:
				#array of the unique values for the given attribute
				attrCategories = self.getAttributeCategories(dataFrame, attribute)
				attrCategories = attrCategories.tolist()
				rares = self.getRares(dataFrame, attribute)
				if len(rares) > 0:
					attrCategories.append("rare")

				for attrCategory in attrCategories:
					if attrCategory in rares:
						continue
					#key = classification, value = probability of P(attr|classification)
					probabilityDict = {}

					#for each of the possible classifications (i.e. 1 or 0)
					for classification in classificationList:
						#skip this case
						if(groundTruth == attribute):
							continue

						if attrCategory == "rare":
							crossAttributeProbability = self.getRareProb(dataFrame, groundTruth, classification, attribute, rares)
						else:
							#the value part of the dictionary: P(a|C)
							crossAttributeProbability = self.calculateCrossAttributeProbability(dataFrame, groundTruth, classification, attribute, attrCategory)

						probabilityDict[classification] = crossAttributeProbability
					#outermost dictionary
					attrDict[attrCategory] = probabilityDict
			model.append(attrDict)

		#Construct a dictionary that will hold the probability of a particular classification C_x (e.g. lieutenant, captain)
		classificationProbabilitiesDict = {}
		#for each of the possible classfications 
		for Cx in classificationList:
			#probability of the particular classification 
			#P = (# people with this particular classification) / (total # of people)
			probOfCx = self.attributeCategoryProbability(dataFrame, dataSet.trueLabels, Cx)
			classificationProbabilitiesDict[Cx] = probOfCx
		#append it to the end of the outermost model array
		model.append(classificationProbabilitiesDict)

	'''
	Pretty prints out the Bayesian model
		dataSet (DataSet) - the dataset
		model (Baysian model object) - the model to print
	'''
	def printModel(self, dataSet, model):
		#Through the outermost model array, we loop up until the 2nd to last element
		#The last element has the dictionary of classification probabilities
		for i in range(0, len(model) - 1):
			print("Attribute: ", dataSet.trainHeaders[i])
			for attrCategory in model[i].keys():
				if(attrCategory == 'mean' or attrCategory == 'std'): #numerical type
					if(attrCategory == 'mean'):
						print("\t Numerical Data: Conditional mean")
					elif(attrCategory == 'std'):
						print("\t Numerical Data: Condition standard deviation")
					for classification in model[i][attrCategory].keys():
						print("\t \t Classification and mean/std: ", classification, ", ", model[i][attrCategory][classification])
				else: #categorical type
					print("\t Attribute Category: ", attrCategory)
					for classification in model[i][attrCategory].keys():
						print("\t \t Classification & Probability: ", classification, ", ", model[i][attrCategory][classification])

		print("Classification Probabilities: ")
		classificationProbs = model[-1]
		for Cx in classificationProbs.keys():
			print("\t Classification: ", Cx)
			print("\t Probability: ", classificationProbs[Cx])

	'''
	Given the attributes of an entry in an dataset and our trained model, classify calculates the P(classification|attributes) 
	for every possible classification, then appends a classification to dataset based on those probabilities. 
	Appends a new column of classifications to the dataset under the header "Bayes Classification" 
		dataSet (DataSet) - the dataset
		testOrTrain (str) - a string that denotes whether we are classifying the train or test set
	Returns: the classified DataFrame
	Note: the testOrTrain parameter exists only because of inheritance; this function will only ever classify the test set.
	'''
	def classify(self, dataSet, testOrTrain):
		dataFrame = dataSet.testDataFrame
		groundTruth = dataSet.trueLabels
		# variable that points to the dictionary of classification probabilities
		classificationList = self.model[-1]
		#make a new column for the data frame where our classifications are going to go
		classificationColumn = []

		#for each of the rows (people) in the dataset
		for row in dataFrame.iterrows():
			#dictionary {key = classification, value = complete bayesian probability}
			bayesianDict = {}
			#dictionary {key = classification, value = numerator probability}
			numeratorDict = {}
			# reset it for every row
			denominatorSum = 0

			#iterate through the possible outcomes of the class variable
			for classification in classificationList.keys():
				#start the numerator product with the value of P(C) for the current classification
				#(we will be multiplying this by all the other attribute probabilities)
				numeratorDict[classification] = classificationList[classification]
				#loop through outer array of the model (but we stop at second to last element of array)
				for j, attributeDict in enumerate(self.model):
					#skip the last element because this isn't an attribute -- it's the classification probabilities dictionary
					if(j == len(self.model) - 1):
						continue
					#if we run into the blank ground truth column, skip this
					if(dataSet.testHeaders[j] == dataSet.trueLabels): #NOTE: this used to be .headers
						continue

					#value for the current row of the given attribute
					attrValue = row[1].iloc[j]
					if(dataSet.testHeaders[j] in dataSet.getNumericalColumns("test")):
						meanDict = attributeDict["mean"]
						stdDict = attributeDict["std"]

						#NUMERATOR = P(person|classification) * P(classification)
						bayesNumerator = self.calculateGaussianProbability(meanDict[classification], stdDict[classification], row[1].iloc[j])
						try:
							numeratorDict[classification] += math.log(bayesNumerator)
						except:
							pass
					else:
						if attrValue in attributeDict:
							bayesNumerator = attributeDict[attrValue][classification]
						else:
							if "rare" in attributeDict:
								bayesNumerator = attributeDict["rare"][classification]
							else:
								bayesNumerator = 1
						try:
							numeratorDict[classification] += math.log(bayesNumerator)
						except:
							pass

			for key in numeratorDict.keys():
				denominatorSum += mpmath.exp(numeratorDict[key] - (max(numeratorDict.items(), key=operator.itemgetter(1))[0]))
			#currently just adding dictionary of all probabilities given all classifications
			#but eventually want to be adding the max of these (the final classification)
			for key in numeratorDict.keys():
				bayesianDict[key] = mpmath.exp(numeratorDict[key] - (max(numeratorDict.items(), key=operator.itemgetter(1))[0])) / denominatorSum

			maxClassification = max(bayesianDict.items(), key=operator.itemgetter(1))[0]
			classificationColumn.append(maxClassification)
		
		#sets new column equal to the array of classifications
		dataFrame["Bayes Classification"] = classificationColumn
		dataSet.resetHeaders(testOrTrain)
		return dataFrame