from Bayes import Bayes
import pandas as pd
import operator

class NaiveBayes(Bayes):

	def __init__(self):
		self.model = [] 

	'''Create model filled as such:
		self.model = array of attributes (e.g. race, position, etc.) where each index points to a dictionary
			attrDict (categorical) { key =  attribute category (e.g. white, black, hispanic), value = probability dictionary}
				probabilityDict = { key = classification (e.g. gets loan, differed, doesn't get loan), value = P(attrCategory|classification)}

		attrDict (numerical) { key = "mean" or "std", value = meanDict or stdDict}
			meanDict = {key = classification, value = conditional mean given this classification}
			stdDict = {key = classification, value = conditional std given this classification}'''
	def train(self, dataSet):

		dataFrame = dataSet.dataFrame
		groundTruth = dataSet.trueLabels
		classificationList = dataFrame[groundTruth].unique()

		#make sure that model has not been classified already 
		if not(dataSet.hasGroundTruth):
			print("Error: Dataset has no ground truth. Cannot train.")
			pass
		#to ensure that we don't train twice
		if bool(self.model):
			print("Error: Model not empty.")
			pass

		#for each of the attributes in the datset (a1...an)
		for attribute in dataSet.headers:
		
			#create outermost dictionary of the model (key = attribute category, value = another dictionary)
			attrDict = {}

			#if numerical type data
			if(attribute in dataSet.getNumericalColumns()):

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

				for attrCategory in attrCategories:

					#key = classification, value = probability of P(attr|classification)
					probabilityDict = {}

					#for each of the possible classifications possible (i.e. lieutenant, captain, etc.)
					for classification in classificationList:
						#skip this case
						if(groundTruth == attribute):
							continue

						#the value part of the dictionary: P(a|C)
						crossAttributeProbability = self.calculateCrossAttributeProbability(dataFrame, groundTruth, classification, attribute, attrCategory)
						probabilityDict[classification] = crossAttributeProbability

					#outermost dictionary
					attrDict[attrCategory] = probabilityDict
			
					
			self.model.append(attrDict)

		print("Model updated!")
		self.printModel(dataSet)


	'''Pretty prints out the Bayesian model '''
	def printModel(self, dataSet):
		for i in range(len(self.model)):
			print("Attribute: ", dataSet.headers[i])
			for attrCategory in self.model[i].keys():
				print("\t Attribute Category: ", attrCategory)
				for classification in self.model[i][attrCategory].keys():
					print("\t \t Classification & Probability: ", classification, ", ", self.model[i][attrCategory][classification])


	'''Given the attributes of an entry in an dataset and our trained model, it calculates the P(classification|attributes) for every
	   possible classification and then appends a classification to dataset based on those probabilities. Appending a new column of classifications
	   to the dataset under the header "Bayes Classification" '''
	def classify(self, dataSet):

		dataFrame = dataSet.dataFrame
		groundTruth = dataSet.trueLabels
		classificationList = dataFrame[groundTruth].unique()

		#make a new column for the data frame where our classifications are going to go
		classificationColumn = []

		#for each of the rows (people) in the dataset
		for row in dataFrame.iterrows():

			#dictionary {key = classification, value = complete bayesian probability}
			bayesianDict = {}
			#dictionary {key = classification, value = numerator probability}
			numeratorDict = {}

			denominatorSum = 0 #reset it for every row

			#iterate through the possible outcomes of the class variable
			for classification in classificationList:

				numeratorDict[classification] = 1

				#loop through outer array of the model
				for j, attributeDict in enumerate(self.model):
					#if we run into the blank ground truth column, skip this row
					if(dataSet.headers[j] == dataSet.trueLabels):
						continue

					#value for the current row of the given attribute
					attrValue = row[1].iloc[j]

					if(dataSet.headers[j] in dataSet.getNumericalColumns()): #numerical
						meanDict = attributeDict["mean"]
						stdDict = attributeDict["std"]
						#P(person|classification) * P(classification)
						bayesNumerator = self.calculateGaussianProbability(meanDict[classification], stdDict[classification], row[1].iloc[j]) * self.attributeCategoryProbability(dataFrame, dataSet.trueLabels, classification)
						numeratorDict[classification] *= bayesNumerator
					else:
						bayesNumerator = attributeDict[attrValue][classification] * self.attributeCategoryProbability(dataFrame, dataSet.trueLabels, classification)
						numeratorDict[classification] *= bayesNumerator

			for key in numeratorDict.keys():
				denominatorSum += numeratorDict[key]
			#currently just adding dictionary of all probabilities given all classifications but eventually want to be adding the max of these (the final classification)
			for key in numeratorDict.keys():
				bayesianDict[key] = round(numeratorDict[key] / denominatorSum, 2)

			classificationColumn.append(bayesianDict)
		
		#sets new column equal to the array of classifications

		dataFrame["Bayes Classification"] = classificationColumn
		print(dataFrame.to_string())
		return dataFrame


		

