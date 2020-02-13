from Bayes import Bayes
from modifiedNaive import ModifiedNaive
import pandas as pd
import operator

class ModifiedBayes(ModifiedNaive):

	'''Instance of modified bayes generates an instance of modified naive bayes that we 
	   can call the train and classify functions on. '''
	def __init__(self):
#		self.nb = ModifiedNaive()
		ModifiedNaive.__init__(self)

	'''Calculates the discrimination score by subtracting the probability of being in the privileged group
	   with a C+ classification minus the probability of being in the underprivileged group in the C+ classification.'''
	def calculateDiscriminationScore(self, CHigherSHigher, CHigherSLower):
		return CHigherSHigher - CHigherSLower

	'''Based on the parameter passed into the modify() function, C+, we manually match up the two possible classifications
	   with the keys "higher" and "lower" inside a dictionary so we can refer to them later.'''
	def assignClassifications(self, classificationDict, CHigher, classesList):
		if (str(classesList[0]) == CHigher):
			classificationDict["higher"] = classesList[0]
			classificationDict["lower"] = classesList[1]
		else:
			classificationDict["higher"] = classesList[1]
			classificationDict["lower"] = classesList[0]

	'''Assigns the keys "higher" and "lower" to the two possible sensitive attribute values based on which of the two has a higher count.
	   S+ ("higher") is the privileged group. We do this based on counts instead of as a manual parameter because there isn't an 'ideal' 
	   sensitive attribute category like there is with classifications.'''
	def assignSensitivity(self, dataSet, dataFrame, sensitivityDict):
		sensitiveAttrCatList = self.getAttributeCategories(dataFrame, dataSet.protectedAttribute)
		Sx = dataFrame.loc[dataFrame[dataSet.protectedAttribute] == sensitiveAttrCatList[0], dataSet.protectedAttribute].count()
		Sy = dataFrame.loc[dataFrame[dataSet.protectedAttribute] == sensitiveAttrCatList[1], dataSet.protectedAttribute].count()
		if (Sx > Sy):
			sensitivityDict["higher"] = sensitiveAttrCatList[0]
			sensitivityDict["lower"] = sensitiveAttrCatList[1]
		else:
			sensitivityDict["higher"] = sensitiveAttrCatList[1]
			sensitivityDict["lower"] = sensitiveAttrCatList[0]

	'''Counts up the number of elements in a particular column that match the classification value located in the classDict passed in 
	   with the key "higher" (AKA - C+).'''
	def calculateNumPos(self, dataFrame, column, classDict):
		return dataFrame.loc[dataFrame[column] == classDict["higher"], column].count()

	'''A function that can be called in the while loop to keep track/ watch how the counts are changing with each iteration'''
	def printCounts(self, dataSet, CHigherSLowerCount, CLowerSLowerCount, CHigherSHigherCount, CLowerSHigherCount, higherOrLowerSensitiveAttributeDict, higherOrLowerClassificationDict):
		dataFrame = dataSet.dataFrame
		print("c+s- count:", CHigherSLowerCount)
		print("c-s- count:", CLowerSLowerCount)
		print("c+s+ count:", CHigherSHigherCount)
		print("c-s+ count:", CLowerSHigherCount)
		print("bayes classification column c+s- count: ", self.countIntersection(dataFrame, dataSet.protectedAttribute, higherOrLowerSensitiveAttributeDict["lower"], "Bayes Classification", higherOrLowerClassificationDict["higher"]))
		print("bayes classification column c-s- count: ", self.countIntersection(dataFrame, dataSet.protectedAttribute, higherOrLowerSensitiveAttributeDict["lower"], "Bayes Classification" , higherOrLowerClassificationDict["lower"]))
		print("bayes classification column c+s+ count: ", self.countIntersection(dataFrame, dataSet.protectedAttribute, higherOrLowerSensitiveAttributeDict["higher"], "Bayes Classification", higherOrLowerClassificationDict["higher"]))
		print("bayes classification column c-s+ count: ", self.countIntersection(dataFrame, dataSet.protectedAttribute, higherOrLowerSensitiveAttributeDict["higher"], "Bayes Classification", higherOrLowerClassificationDict["lower"]))

	'''Space saving function for modify() that prints out probabilities'''
	def printProbabilities(self, CHigherSLower, CLowerSLower, CHigherSHigher, CLowerSHigher):
		print("c+s- prob:", CHigherSLower)
		print("c-s- prob:", CLowerSLower)
		print("c+s+ prob:", CHigherSHigher)
		print("c-s+ prob:", CLowerSHigher)

		
	def train(self, dataSet, CHigher):
		ModifiedNaive.train(self, dataSet, self.model)
		self.modify(dataSet, CHigher)
		
		
		
	'''Trains and classifies the dataset '''
	def modify(self, dataSet, CHigher):
		dataFrame = dataSet.trainDataFrame
		protected = dataSet.protectedAttribute
		groundTruth = dataSet.trueLabels
		sensitiveAttributeModelIndex = dataSet.trainHeaders.index(protected) #need to know index of sensitive attribute in the model

		dataFrame = self.classify(dataSet, "train")

		#Assign dictionary values based on CHigher parameter
		classesList = self.getAttributeCategories(dataFrame, dataSet.trueLabels)
		higherOrLowerClassificationDict = {}
		self.assignClassifications(higherOrLowerClassificationDict, CHigher, classesList)

		#Assign the two sensitive attribute categories as S+ and S-
		higherOrLowerSensitiveAttributeDict = {}
		self.assignSensitivity(dataSet, dataFrame, higherOrLowerSensitiveAttributeDict)

		#calculate the number of people in the dataset that are actually classified as C+ (in the ground truth column - the real number from the data)
		actualNumPos = self.calculateNumPos(dataFrame, groundTruth, higherOrLowerClassificationDict)

		#Compute counts for C+S-,C-S+,C+S+,and C-S- based on counts from the original groundTruth column
		CHigherSLowerCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"], groundTruth, higherOrLowerClassificationDict["higher"])
		CLowerSHigherCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"], groundTruth , higherOrLowerClassificationDict["lower"])
		CHigherSHigherCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"],groundTruth, higherOrLowerClassificationDict["higher"])
		CLowerSLowerCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"], groundTruth, higherOrLowerClassificationDict["lower"])
		#Compute baseline probabilities based on the corresponding counts above, which will be used to calculate the preliminary disc score
		CHigherSLower = CHigherSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])
		CHigherSHigher = CHigherSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
		CLowerSLower = CLowerSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])
		CLowerSHigher = CLowerSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
		# self.printProbabilities(CHigherSLower, CLowerSLower, CHigherSHigher, CLowerSHigher)
		
		#Calculate the preliminary discrimination score -- disc = P(C+ | S+) - P(C+ | S-)
		disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)

		while (disc > 0.0):
			#Calculate numPos -- the number of instances that we classify people as C+
			numPos = self.calculateNumPos(dataFrame, "Bayes Classification", higherOrLowerClassificationDict)
			
			weightOfChange = 0.01 #Value by which we will be modifiying the counts

			#Uncomment if desired: prints out current artificial counts we're modifiying and current actual counts in bayes classification column
			#self.printCounts(dataSet, CHigherSLowerCount, CLowerSLowerCount, CHigherSHigherCount, CLowerSHigherCount, higherOrLowerSensitiveAttributeDict, higherOrLowerClassificationDict)

			if (numPos < actualNumPos): #We have more positive C+ labels we can assign

				#Slightly increase the count for C+S- and slightly decrease the count for C-S-
				CHigherSLowerCount = CHigherSLowerCount + (weightOfChange * CLowerSHigherCount)
				CLowerSLowerCount = CLowerSLowerCount - (weightOfChange * CLowerSHigherCount)

				#Update the probabilities based on these new counts
				CHigherSLower = CHigherSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])
				CLowerSLower = CLowerSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])

				#Overwrite the old probabilities in the model
				self.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]] = CHigherSLower
				self.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]] = CLowerSLower

			else: #we have assigned more positive C+ labels than we should be
			
				#Slightly increase the count for the C-S+ and slightly decrease the count for C+S+ 
				CLowerSHigherCount = CLowerSHigherCount + (weightOfChange * CHigherSLowerCount)
				CHigherSHigherCount = CHigherSHigherCount - (weightOfChange * CHigherSLowerCount)

				#Update the probabilities based on these new counts
				CLowerSHigher = CLowerSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
				CHigherSHigher = CHigherSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
				
				#Overwrite the old probabilities in the model
				self.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]] = CLowerSHigher
				self.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]] = CHigherSHigher

			
			#reclassify and recompute the new discrimination score
			dataFrame = self.classify(dataSet, "train")
			disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)
			# self.printProbabilities(CHigherSLower, CLowerSLower, CHigherSHigher, CLowerSHigher)

		#print out the final classifications
		#print(dataFrame.to_string())
		'''Uncomment if desired: Call to save classifications to a csv file called modifiedBayesClassifications.csv'''
		#dataFrame.to_csv('modifiedBayesClassification.csv', sep='\t', encoding='utf-8')



