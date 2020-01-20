from Bayes import Bayes
from modifiedNaive import ModifiedNaive
import pandas as pd
import operator

class ModifiedBayes(Bayes):

	'''Instance of modified bayes generates an instance of naive bayes that we 
	   can call naiveBayes functions on. '''
	def __init__(self, ds, fileName, protectedAttribute, trueLabel):
		ds.loadData(fileName, [protectedAttribute], trueLabel)
		self.nb = ModifiedNaive()

	def calculateDiscriminationScore(self, CHigherSHigher, CHigherSLower):
		return CHigherSHigher - CHigherSLower

	'''A test function to compute the # of values that have changed from the original dataset'''
	def testVals(self, dataSet, CPlus, CMinus, SPlus, SMinus):
		print("c+s+ count original: ", self.countIntersection(dataSet.dataFrame, dataSet.trueLabels, CPlus, dataSet.protectedAttributes[0], SPlus ))
		print("c-s+ count original: ", self.countIntersection(dataSet.dataFrame, dataSet.trueLabels, CMinus, dataSet.protectedAttributes[0], SPlus ))
		print("c-s- count original: ", self.countIntersection(dataSet.dataFrame, dataSet.trueLabels, CMinus, dataSet.protectedAttributes[0], SMinus ))
		print("c+s- count original: ", self.countIntersection(dataSet.dataFrame, dataSet.trueLabels, CPlus, dataSet.protectedAttributes[0], SMinus ))

		print("new number C+S+:", self.countIntersection(dataSet.dataFrame, "Bayes Classification", CPlus, dataSet.protectedAttributes[0], SPlus ))
		print("new number C-S+:", self.countIntersection(dataSet.dataFrame, "Bayes Classification", CMinus, dataSet.protectedAttributes[0], SPlus ))
		print("new number C-S-:", self.countIntersection(dataSet.dataFrame, "Bayes Classification", CMinus, dataSet.protectedAttributes[0], SMinus))
		print("new number C+S-:", self.countIntersection(dataSet.dataFrame, "Bayes Classification", CPlus, dataSet.protectedAttributes[0], SMinus ))


	def modify(self, dataSet, CHigher):

		dataFrame = dataSet.dataFrame
		self.nb.train(dataSet)
		dataFrame = self.nb.classify(dataSet)
		print("original naive bayes classifications")

		protected = dataSet.protectedAttributes[0]
		groundTruth = dataSet.trueLabels

		#assign dictionary values  based on CHigher parameter
		classesList = self.getAttributeCategories(dataFrame, dataSet.trueLabels)
		higherOrLowerClassificationDict = {}
		if (str(classesList[0]) == CHigher):
			higherOrLowerClassificationDict["higher"] = classesList[0]
			higherOrLowerClassificationDict["lower"] = classesList[1]
		else:
			higherOrLowerClassificationDict["higher"] = classesList[1]
			higherOrLowerClassificationDict["lower"] = classesList[0]

		#calculate the number of people in the dataset that are actually classified as C+
		actualNumPos = dataFrame.loc[dataFrame[groundTruth] == higherOrLowerClassificationDict["higher"], groundTruth].count()
		print("actual num pos", actualNumPos)

		#figure out which of the two sensitive attribute categories should be S+ and S-
		sensitiveAttrCatList = self.getAttributeCategories(dataFrame, dataSet.protectedAttributes[0])
		higherOrLowerSensitiveAttributeDict = {}
		Sx = dataFrame.loc[dataFrame[dataSet.protectedAttributes[0]] == sensitiveAttrCatList[0], dataSet.protectedAttributes[0]].count()
		Sy = dataFrame.loc[dataFrame[dataSet.protectedAttributes[0]] == sensitiveAttrCatList[1], dataSet.protectedAttributes[0]].count()
		if (Sx > Sy):
			higherOrLowerSensitiveAttributeDict["higher"] = sensitiveAttrCatList[0]
			higherOrLowerSensitiveAttributeDict["lower"] = sensitiveAttrCatList[1]
		else:
			higherOrLowerSensitiveAttributeDict["higher"] = sensitiveAttrCatList[1]
			higherOrLowerSensitiveAttributeDict["lower"] = sensitiveAttrCatList[0]


		print(higherOrLowerClassificationDict)
		print(higherOrLowerSensitiveAttributeDict)
		#We need to know which index the sensitive attribute is located at in the model
		#Good news is that we're only going to have to do this once - we could potentially ask the data team to add this feature
		for i in range (len(self.nb.model) -1):
			if(dataSet.headers[i] == dataSet.protectedAttributes[0]):
				sensitiveAttributeModelIndex = i #save this index
		

		CHigherSLowerCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"], "Bayes Classification", higherOrLowerClassificationDict["higher"])
		CLowerSHigherCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"], "Bayes Classification" , higherOrLowerClassificationDict["lower"])
		CHigherSHigherCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"], "Bayes Classification", higherOrLowerClassificationDict["higher"])
		CLowerSLowerCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"], "Bayes Classification", higherOrLowerClassificationDict["lower"])

		CHigherSLower = CHigherSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])
		CLowerSLower = CLowerSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])
		CLowerSHigher = CLowerSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
		CHigherSHigher = CHigherSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])

		#Calculate the preliminary/baseline discrimination score
		#   disc = P(C+ | S+) - P(C+ | S-)
		disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)
		print("Original discrimination score: ", disc)
		
		print("original counts")
		print("c+s- count:", CHigherSLowerCount)
		print("c-s- count:", CLowerSLowerCount)
		print("c+s+ count:", CHigherSHigherCount)
		print("c-s+ count:", CLowerSHigherCount)
		
		while (disc > 0.0):

			# #every time we get into an iteration of the while loop we want to grab these values from the model so we have the updated probabilities
			# #also because for the first iteration we want it to grab the original probabilities
			# CLowerSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]]
			# CHigherSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]]
			# CLowerSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]]
			# CHigherSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]]

			print(" ")
			print("C-S+", CLowerSHigher)
			print("C+S-", CHigherSLower)
			print("C-S-", CLowerSLower)
			print("C+S+", CHigherSHigher)

			#Calculate numPos (the number of instances that we classify people as C+)
			numPos = dataFrame.loc[dataFrame["Bayes Classification"] == higherOrLowerClassificationDict["higher"], "Bayes Classification"].count()
			print("numPos is:", numPos)
			
			weightOfChange = 0.03

			print("c+s- count:", CHigherSLowerCount)
			print("c-s- count:", CLowerSLowerCount)
			print("c+s+ count:", CHigherSHigherCount)
			print("c-s+ count:", CLowerSHigherCount)

			#print("model at the beginning of the while loop iteration")
			#self.nb.printModel(dataSet)

			if (numPos < actualNumPos): #We have more positive labels we can assign
				countTemp = CHigherSLowerCount
				
				CHigherSLowerCount = CHigherSLowerCount + (weightOfChange * CLowerSHigherCount)
				CLowerSLowerCount = countTemp - (weightOfChange * CLowerSHigherCount)

				CHigherSLower = CHigherSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])
				CLowerSLower = CLowerSLowerCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"])


				#model[attribute][S+ or S-][C- or C+]
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]] = CHigherSLower #C+S-
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]] = CLowerSLower #C-S-

			else: #we have assigned more positive labels than we should be

				countTemp = CLowerSHigherCount
			
				CLowerSHigherCount = CLowerSHigherCount + (weightOfChange * CHigherSLowerCount)
				CHigherSHigherCount = countTemp - (weightOfChange * CHigherSLowerCount)

				CLowerSHigher = CLowerSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
				CHigherSHigher = CHigherSHigherCount / self.countAttr(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"])
				
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]] = CLowerSHigher #C-S+
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]] = CHigherSHigher #C+S+



			dataFrame = self.nb.classify(dataSet)
			disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)
			print("disc at the end of the iteration: ", disc)
			# self.nb.printModel(dataSet) #print the model at the end of the iteration
			# print(dataFrame.to_string())
			

		print("finished")
		#print out the final classifications
		print(dataFrame.to_string())
		print("COMPARING RESULTS:\n")
		self.testVals(dataSet, higherOrLowerClassificationDict["higher"], higherOrLowerClassificationDict["lower"], higherOrLowerSensitiveAttributeDict["higher"], higherOrLowerSensitiveAttributeDict["lower"])




