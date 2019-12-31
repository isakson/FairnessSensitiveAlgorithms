from Bayes import Bayes
from NaiveBayes import NaiveBayes
import pandas as pd
import operator

class ModifiedBayes(Bayes):

	'''Instance of modified bayes generates an instance of naive bayes that we 
	   can call naiveBayes functions on. '''
	def __init__(self, ds, fileName, protectedAttribute, trueLabel):
		ds.loadData(fileName, [protectedAttribute], trueLabel)
		self.nb = NaiveBayes()

	def calculateDiscriminationScore(self, CHigherSHigher, CHigherSLower):
		return CHigherSHigher - CHigherSLower

	def modify(self, dataSet):

		dataFrame = dataSet.dataFrame
		self.nb.train(dataSet)
		self.nb.classify(dataSet)

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

		#Calculate the discrimination score
		#   disc = P(C+ | S+) - P(C+ | S-) 

		CHigherSHigher = self.calculateCrossAttributeProbability(dataFrame, dataSet.protectedAttributes[0], higherOrLowerSensitiveAttributeDict["higher"], dataSet.trueLabels, higherOrLowerClassificationDict["higher"])
		#temporary variable we'll use to calculate the preliminary discrimination score without modifiying CHigherSLower
		NewCHigherSLower = self.calculateCrossAttributeProbability(dataFrame, dataSet.protectedAttributes[0], higherOrLowerSensitiveAttributeDict["lower"], dataSet.trueLabels, higherOrLowerClassificationDict["higher"])
		disc = self.calculateDiscriminationScore(CHigherSHigher, NewCHigherSLower)
		print("Original discrimination score: ", disc)

		#Calculate numPos (the number of instances that we classify people as C+)
		print("protected attribute: ", dataSet.protectedAttributes[0])
		numPos = dataFrame.loc[dataFrame["Bayes Classification"] == higherOrLowerClassificationDict["higher"], "Bayes Classification"].count()
		print("FIRST numPos:", numPos)

		#We need to know which index the sensitive attribbute is located at in the model
		#Good news is that we're only going to have to do this once - we could potentially ask the data team to add this feature
		for i in range (len(self.nb.model) -1):
			if(dataSet.headers[i] == dataSet.protectedAttributes[0]):
				sensitiveAttributeModelIndex = i #save this index

		while (disc > 0.0):

			#every time we get into an iteration of the while loop we want to grab these values from the model so we have the updated probabilities
			#also because for the first iteration we want it to grab the original probabilities
			CLowerSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]]
			CHigherSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]]
			CLowerSLower = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]]
			CHigherSHigher = self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]]
			
			weightOfChange = 0.01

			if (numPos < actualNumPos): #We have more positive labels we can assign
				CHigherSLower = CHigherSLower + (weightOfChange * CLowerSHigher)
				CLowerSLower = CHigherSLower - (weightOfChange * CLowerSHigher)
				#model[attribute][S+ or S-][C- or C+]
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["higher"]] = CHigherSLower #C+S-
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["lower"]][higherOrLowerClassificationDict["lower"]] = CLowerSLower #C-S-

			else: #we have assigned more positive labels than we should be
				CLowerSHigher = CLowerSHigher + (weightOfChange * CHigherSLower)
				CHigherSHigher = CLowerSHigher - (weightOfChange * CHigherSLower)
				#model[attribute][S+ or S-][C- or C+]
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["lower"]] = CLowerSHigher #C-S+
				self.nb.model[sensitiveAttributeModelIndex][higherOrLowerSensitiveAttributeDict["higher"]][higherOrLowerClassificationDict["higher"]] = CHigherSHigher #C+S+

			for C in classesList:
				#Essentially putting the last iteration's P(c)'s  into the model
				probOfC = self.attributeCategoryProbability(dataFrame, dataSet.trueLabels, C)
				self.nb.model[-1][C] = probOfC

			disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)

			print("disc at the end of the iteration: ", disc)
			self.nb.printModel(dataSet) #print the model at the end of the iteration
			self.nb.classify(dataSet)

			#recompute the new numPos
			numPos = dataFrame.loc[dataFrame["Bayes Classification"] == higherOrLowerClassificationDict["higher"], "Bayes Classification"].count()
			print("new num pos: ", numPos)

		print("finished")
		#print out the final classifications
		print(dataFrame.to_string())
				
		#Important questions
		'''the two things that we are modifiying in each case are not the things that should sum to 1, therefore, should we be also modifying the other 2, 
	   	   not by increasing/decreasing probabilities, but just by calculating the cross attribute probability of the two? - otherwise things will not necessarily be adding to 1.
	       This is because the two things that are supposed to add to 1 are based on the classifications (S+C- and S-C- should add to 1) - is this the same for 
	       C-S+ and C-S-?
		'''		



