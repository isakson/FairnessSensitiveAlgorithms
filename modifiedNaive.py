from Bayes import Bayes
import pandas as pd
import operator

class ModifiedNaive(Bayes):

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
		protected = dataSet.protectedAttribute

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
						#if the attribute is the sensitive attribute then calculate P(C|S) instead of P(a|C)
						if(attribute == protected):
							crossAttributeProbability = self.calculateCrossAttributeProbability(dataFrame, attribute, attrCategory, groundTruth, classification)
						else:
							crossAttributeProbability = self.calculateCrossAttributeProbability(dataFrame, groundTruth, classification, attribute, attrCategory)
						probabilityDict[classification] = crossAttributeProbability

					#outermost dictionary
					attrDict[attrCategory] = probabilityDict
			
					
			self.model.append(attrDict)

		#Construct a dictionary that will hold the probability of each sensitive attribute S_x (e.g. male, female)
		sensitiveProbabilitiesDict = {}
		#for each of the sensitive attributes 
		for Sx in dataFrame[dataSet.protectedAttribute].unique():
			#P = (# people belonging to this sensitive group) / (total # of people)
			probOfSx = self.attributeCategoryProbability(dataFrame, dataSet.protectedAttribute, Sx)
			sensitiveProbabilitiesDict[Sx] = probOfSx
		#append it to the end of the outermost model array
		self.model.append(sensitiveProbabilitiesDict)

		print("\nMODEL UPDATED... PRINTING MODEL...!\n")
		self.printModel(dataSet)
		print("\n FINISHED PRINTING MODEL. \n")


	'''Pretty prints out the Bayesian model '''
	def printModel(self, dataSet):
		#Through the outermost model array, we loop up until the 2nd to last element
		for i in range(0, len(self.model) - 1):
			print("Attribute: ", dataSet.headers[i])
			for attrCategory in self.model[i].keys():
				if(attrCategory == 'mean' or attrCategory == 'std'): #numerical type
					if(attrCategory == 'mean'):
						print("\t Numerical Data: Conditional mean")
					elif(attrCategory == 'std'):
						print("\t Numerical Data: Condition standard deviation")
					for classification in self.model[i][attrCategory].keys():
						print("\t \t Classification and mean/std: ", classification, ", ", self.model[i][attrCategory][classification])
				else: #categorical type
					print("\t Attribute Category: ", attrCategory)
					for classification in self.model[i][attrCategory].keys():
						print("\t \t Classification & Probability: ", classification, ", ", self.model[i][attrCategory][classification])

		print("Classification Probabilities: ")
		classificationProbs = self.model[-1]
		for Cx in classificationProbs.keys():
			print("\t Classification: ", Cx)
			print("\t Probability: ", classificationProbs[Cx])

	'''Given the attributes of an entry in an dataset and our trained model, it calculates the P(classification|attributes) for every
	   possible classification and then appends a classification to dataset based on those probabilities. Appending a new column of classifications
	   to the dataset under the header "Bayes Classification" '''
	def classify(self, dataSet):

		dataFrame = dataSet.dataFrame
		groundTruth = dataSet.trueLabels

		classificationList = dataFrame[groundTruth].unique()
		sensitiveList = self.model[-1] #variable that points to the dictionary of classification probabilities

		#make a new column for the data frame where our classifications are going to go
		classificationColumn = []

		#for each of the rows (people) in the dataset
		for row in dataFrame.iterrows():

			#dictionary {key = classification, value = complete bayesian probability}
			bayesianDict = {}
			#dictionary {key = classification, value = numerator probability}
			numeratorDict = {}
			denominatorSum = 0 #reset it for every row

			#Get the person's sensitive group
			ind = dataSet.headers.index(dataSet.protectedAttribute)
			sensitiveGroup = row[1].iloc[ind]

			#iterate through the possible outcomes of the class variable
			for classification in classificationList:

				numeratorDict[classification] = sensitiveList[sensitiveGroup]

				#loop through outer array of the model (but we stop at second to last element of array)
				for j, attributeDict in enumerate(self.model):

					#skip the last element because this isn't an attribute -- it's the classification probabilities dictionary
					if(j == len(self.model) - 1):
						continue
					#if we run into the blank ground truth column, skip this
					if(dataSet.headers[j] == dataSet.trueLabels):
						continue

					#value for the current row of the given attribute
					attrValue = row[1].iloc[j]

					#NUMERATOR = the product of P(a1|C)...P(an|C)*P(S)
					if(dataSet.headers[j] in dataSet.getNumericalColumns()): #numerical
						meanDict = attributeDict["mean"]
						stdDict = attributeDict["std"]

						bayesNumerator = self.calculateGaussianProbability(meanDict[classification], stdDict[classification], row[1].iloc[j])
						numeratorDict[classification] *= bayesNumerator
					else:
						bayesNumerator = attributeDict[attrValue][classification]
						numeratorDict[classification] *= bayesNumerator

			#add together probabilities for each classification so we can divide each of them by the sum to normalize them
			for key in numeratorDict.keys():
				denominatorSum += numeratorDict[key]
			for key in numeratorDict.keys():
				bayesianDict[key] = round(numeratorDict[key] / denominatorSum, 2)

			maxClassification = max(bayesianDict.items(), key=operator.itemgetter(1))[0]
			classificationColumn.append(maxClassification)
		
		#sets new column equal to the array of classifications
		dataFrame["Bayes Classification"] = classificationColumn
		
		#print(dataFrame.to_string())
		return dataFrame


		

