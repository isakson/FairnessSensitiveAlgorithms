from Bayes import Bayes
import pandas as pd
import operator

class NaiveBayes(Bayes):

	def __init__(self):
		#we need to handle the fact that when we remove the case where ground truth = attribute we remove this probability so the indeces aren't going to match
		self.model = [] 
		# dictionary of arrays of dictionaries
		# outermost dict = keys: each possible classification, value: probabilistic model for that classification (array)
		# array = list of the attribute dictionaries
		# inner dict = key: particular attribute category, value = probability of (a|C)
		# inner dict (numerical)= key: "mean" or "std", value = mean or std

		#outermost array of all the attributes, each of which points to a dictionary
		#outermost dictionary = {attributeCategory : {<C+, probability>, <C_, probability>} }}


	def train(self, dataSet):

		dataFrame = dataSet.dataFrame
		groundTruth = dataSet.trueLabels[0]

		if not(dataSet.hasGroundTruth):
			print("Error: Dataset has no ground truth. Cannot train.")
			pass
		if bool(self.model):
			print("Error: Model not empty.")
			pass

		#for each of the attributes in the datset (a1...an)

		for attribute in dataSet.headers:
		
			attrDict = {}

			if(attribute in dataSet.getNumericalColumns()):
				attrCategories = self.getAttributeCategories(dataFrame, attribute)
				#for each of the possible categorizations of an attribute
				for attrCategory in attrCategories:
					meanDict = {}
					stdDict = {}

					#for each of the possible classifiactions possible (i.e. lieutenant, captain, etc.)
					for category in dataFrame[groundTruth].unique():
						if(groundTruth == attribute):
							continue

						mean = self.calculateConditionalMean(dataFrame, attribute, groundTruth, category)
						std = self.calculateConditionalStandardDeviation(dataFrame, attribute, groundTruth, category)
						meanDict[category] = mean
						stdDict[category] = std
					attrDict["mean"] = meanDict
					attrDict["std"] = stdDict

			
			else:
				attrCategories = self.getAttributeCategories(dataFrame, attribute)
				for attrCategory in attrCategories:
					probabilityDict = {}

					#for each of the possible classifiactions possible (i.e. lieutenant, captain, etc.)
					
					for category in dataFrame[groundTruth].unique():
						if(groundTruth == attribute):
							continue
						categoryProb = self.calculateCrossAttributeProbability(dataFrame, groundTruth, category, attribute, attrCategory)
						probabilityDict[attrCategory] = categoryProb

					attrDict[attrCategory] = probabilityDict
			
					
			self.model.append(attrDict)

		print("Model updated!")
		self.printModel(dataSet)


	'''a function for testing that prints out the model generated '''
	def printModel(self, dataSet):

		for i in range(len(self.model)):
			print("Attribute: ", dataSet.headers[i])

			for attrCategory in self.model[i].keys():
				print("\t Attribute Category: ", attrCategory)

				for classification in self.model[i][attrCategory].keys():
					print("\t \t Classification & Probability: ", classification, ", ", self.model[i][attrCategory][classification])


	def classify(self, dataSet):

		dataFrame = dataSet.dataFrame
		groundTruth = dataSet.trueLabels[0]

		#update data set
		if dataSet.hasGroundTruth:
			print("Error: Dataset already is classified.")
			pass
			#Dont think pass is what we want for an error (because pass doesn't do anything)


			#should we remove the true labels 0 index


		#make a new column for the data frame where our classifications are going to go
		classificationColumn = []

		#for each of the rows (people) in the dataset
		for row in dataFrame.iterrows():

			classCategories = {}

			#iterate through the model array
			for j, attribute in enumerate(self.model):

				#print(dataSet.headers[j])
				print (j)
				print(row[1])
				print("series index", row[1].iloc[j])
				#loc is for label based indexing

				#if we run into the blank ground truth column, skip this row
				if(row[1].iloc[j] == '***'):
					continue

				denominatorSum = 0
				probabilityDict = {} #key = class category, value = numerator (p(d|h)* p(h))

				if(attribute in dataSet.getNumericalColumns()): #numerical

					meanDict = atttribute["mean"]
					stdDict = attribute["std"]
					for classCategory in meanDict.keys():
						P = self.caclulateGaussianProbability(meanDict[classCategory], stdDict[classCategory], row[j]) * self.attributeCategoryProbability(dataFrame, trueLabels[0], classCategory)
						probabilityDict[classCategory] = P
						denominatorSum += P

				else:
					currDict = row[1].iloc[j]

					for classCategory in currDict.keys():
						P = currDict[classCategory] * self.attributeCategoryProbability(dataFrame, trueLabels[0], classCategory)
						probabilityDict[classCategory] = P
						denominatorSum += P
				
				for numerator in probabilityDict.keys():

					#if it's the first time we're seeing this classification
					if not numerator in classCategories:
						classCategories[numerator] = 1

					classCategories[numerator] *= probabilityDict[numerator] / denominatorSum 

			print("cat", classCategories)

			classificationColumn.append(max(classCategories, key = classCategories.get))
		
		#sets new column equal to the array of classifications
		dataFrame["classification"] = classificationColumn
		return dataFrame


		

