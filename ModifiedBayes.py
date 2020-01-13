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

	def modify(self, dataSet):

		dataFrame = dataSet.dataFrame
		self.nb.train(dataSet)
		dataFrame = self.nb.classify(dataSet)
		protected = dataSet.protectedAttributes[0]
		groundTruth = dataSet.trueLabels

		#figure out which of the two possible classifications should be C+ and C-
		classesList = self.getAttributeCategories(dataFrame, dataSet.trueLabels)
		higherOrLowerClassificationDict = {}
		Cx = dataFrame.loc[dataFrame[dataSet.trueLabels] == classesList[0], dataSet.trueLabels].count()
		Cy = dataFrame.loc[dataFrame[dataSet.trueLabels] == classesList[1], dataSet.trueLabels].count()
		if (Cx > Cy):
			higherOrLowerClassificationDict["higher"] = classesList[0]
			higherOrLowerClassificationDict["lower"] = classesList[1]
		else:
			higherOrLowerClassificationDict["higher"] = classesList[1]
			higherOrLowerClassificationDict["lower"] = classesList[0]


		#calculate the number of people in the dataset that are actually classified as C+
		actualNumPos = dataFrame.loc[dataFrame[dataSet.trueLabels] == higherOrLowerClassificationDict["higher"], dataSet.trueLabels].count()
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


		#We need to know which index the sensitive attribute is located at in the model
		#Good news is that we're only going to have to do this once - we could potentially ask the data team to add this feature
		for i in range (len(self.nb.model) -1):
			if(dataSet.headers[i] == dataSet.protectedAttributes[0]):
				sensitiveAttributeModelIndex = i #save this index

		print("PRINTING BC COLUMN #########################")
		print(dataFrame["Bayes Classification"])


		#Calculate the preliminary/baseline discrimination score
		#   disc = P(C+ | S+) - P(C+ | S-) 
		CHigherSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]]
		CHigherSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]]
		disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)
		print("Original discrimination score: ", disc)
		
		while (disc > 0.0):

			#every time we get into an iteration of the while loop we want to grab these values from the model so we have the updated probabilities
			#also because for the first iteration we want it to grab the original probabilities
			CLowerSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]]
			CHigherSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]]
			CLowerSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]]
			CHigherSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]]

			print("C-S+", CLowerSHigher)
			print("C+S-", CHigherSLower)
			print("C-S-", CLowerSLower)
			print("C+S+", CHigherSHigher)

			#Calculate numPos (the number of instances that we classify people as C+)
			numPos = dataFrame.loc[dataFrame["Bayes Classification"] == higherOrLowerClassificationDict["higher"], "Bayes Classification"].count()
			print("numPos is:", numPos)
			
			weightOfChange = 0.01

			print("model at the beginning of the while loop iteration")
			self.nb.printModel(dataSet)

			if (numPos < actualNumPos): #We have more positive labels we can assign

				CHigherSLowerCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"], "Bayes Classification", higherOrLowerClassificationDict["higher"])
				CLowerSHigherCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"], "Bayes Classification" , higherOrLowerClassificationDict["lower"])

				CHigherSLowerCount = CHigherSLowerCount + (weightOfChange * CLowerSHigherCount)
				CLowerSLowerCount = CHigherSLowerCount - (weightOfChange * CLowerSHigherCount)

				CHigherSLower = (CHigherSLowerCount / len(dataFrame.index)) / self.attributeCategoryProbability(dataFrame, groundTruth, higherOrLowerClassificationDict["higher"])
				CLowerSLower = (CLowerSLowerCount / len(dataFrame.index)) / self.attributeCategoryProbability(dataFrame, groundTruth, higherOrLowerClassificationDict["lower"])
				
				print("new probability chigherslower", CHigherSLower)
				print("new probability clowerslower", CLowerSLower)

				#model[attribute][S+ or S-][C- or C+]
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]] = CHigherSLower #C+S-
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]] = CLowerSLower #C-S-

			else: #we have assigned more positive labels than we should be
				CHigherSLowerCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["lower"], "Bayes Classification", higherOrLowerClassificationDict["higher"])
				CLowerSHigherCount = self.countIntersection(dataFrame, protected, higherOrLowerSensitiveAttributeDict["higher"], "Bayes Classification", higherOrLowerClassificationDict["lower"])

				CLowerSHigherCount = CLowerSHigherCount + (weightOfChange * CHigherSLowerCount)
				CHigherSHigherCount = CLowerSHigherCount - (weightOfChange * CHigherSLowerCount)

				CLowerSHigher = (CLowerSHigherCount / len(dataFrame.index)) / self.attributeCategoryProbability(dataFrame, groundTruth, higherOrLowerClassificationDict["lower"])
				CHigherSHigher = (CHigherSHigherCount / len(dataFrame.index)) / self.attributeCategoryProbability(dataFrame, groundTruth, higherOrLowerClassificationDict["higher"])

				
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]] = CLowerSHigher #C-S+
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]] = CHigherSHigher #C+S+



			dataFrame = self.nb.classify(dataSet)
			disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)
			print("disc at the end of the iteration: ", disc)
			self.nb.printModel(dataSet) #print the model at the end of the iteration
			#print(dataFrame.to_string())
			

		print("finished")
		#print out the final classifications
		print(dataFrame.to_string())
				
		#Important questions
		'''the two things that we are modifiying in each case are not the things that should sum to 1, therefore, should we be also modifying the other 2, 
	   	   not by increasing/decreasing probabilities, but just by calculating the cross attribute probability of the two? - otherwise things will not necessarily be adding to 1.
	       This is because the two things that are supposed to add to 1 are based on the classifications (S+C- and S-C- should add to 1) - is this the same for 
	       C-S+ and C-S-?
		'''		



