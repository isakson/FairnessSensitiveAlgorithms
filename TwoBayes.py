from NaiveBayes import NaiveBayes
from Bayes import Bayes
from DataSet import DataSet
from ModifiedBayes import ModifiedBayes
import operator
from modifiedNaive import ModifiedNaive
import math
import mpmath

class TwoBayes(NaiveBayes, ModifiedBayes):

	def __init__(self):
		self.modelX = []
		self.modelY = []
		self.Sx = ""
		self.Sy = ""

	'''
	Assigns the keys "higher" and "lower" to the two possible sensitive attribute values based on which of the two 
	has a higher count. S+ ("higher") is the privileged group. We do this based on counts instead of as a manual 
	parameter because there isn't an 'ideal' sensitive attribute category like there is with classifications.
		dataSet (DataSet) - the dataset
	'''
	def assignSensitivity(self, dataSet):
		dataFrame = dataSet.trainDataFrame
		sensitiveAttrCatList = self.getAttributeCategories(dataFrame, dataSet.protectedAttribute)
		Sa = dataFrame.loc[dataFrame[dataSet.protectedAttribute] == sensitiveAttrCatList[0], dataSet.protectedAttribute].count()
		Sb = dataFrame.loc[dataFrame[dataSet.protectedAttribute] == sensitiveAttrCatList[1], dataSet.protectedAttribute].count()
		if (Sa > Sb):
			self.Sx = sensitiveAttrCatList[0]
			self.Sy = sensitiveAttrCatList[1]
		else:
			self.Sx = sensitiveAttrCatList[1]
			self.Sy = sensitiveAttrCatList[0]

	'''
	Splits the train dataframe into two dataframes, grouped by the sensitive attribute
		dataSet (DataSet) - the dataset
		sensitiveVal (str) - the value of the protected attribute label
	Returns: the dataset
	'''
	def splitDataFrame(self, dataSet, sensitiveVal):
		sensitiveAttr = dataSet.protectedAttribute
		df = dataSet.trainDataFrame
		ds = DataSet()
		ds.fileName = dataSet.fileName
		ds.protectedAttribute = sensitiveAttr
		ds.trueLabels = dataSet.trueLabels
		ds.trainHeaders = dataSet.trainHeaders
		ds.numAttributes = dataSet.numAttributes

		try:
			ds.trainDataFrame = df.groupby([sensitiveAttr]).get_group(sensitiveVal)
		except:
			return 0
		return ds

	'''
	Trains the model.
		dataSet (DataSet) - the dataset
		CHigher (str) - C+
	'''
	def train(self, dataSet, CHigher):
		self.assignSensitivity(dataSet)
		dsX = self.splitDataFrame(dataSet, self.Sx)
		dsY = self.splitDataFrame(dataSet, self.Sy)

		NaiveBayes.train(self, dsX, self.modelX)
		NaiveBayes.train(self, dsY, self.modelY)

		self.modify(dataSet, CHigher)

	'''
	Given the attributes of an entry in an dataset and our trained model, classify calculates the P(classification|attributes)
	for every possible classification and then appends a classification to the dataset based on those probabilities. 
	Appends a new column of classifications to the dataset under the header "Bayes Classification" 
		dataSet (DataSet) - the dataset
		testOrTrain (str) - a string indicating whether we are classifying the train or testing set
	'''
	def classify(self, dataSet, testOrTrain):
		if testOrTrain == "test":
			dataFrame = dataSet.testDataFrame
			headers = dataSet.testHeaders
		else:
			dataFrame = dataSet.trainDataFrame
			headers = dataSet.trainHeaders
		groundTruth = dataSet.trueLabels
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
			ind = headers.index(dataSet.protectedAttribute)
			sensitiveGroup = row[1].iloc[ind]
			currModel = self.modelY
			classificationList = self.modelY[-1]
			if (sensitiveGroup == self.Sx):
				currModel = self.modelX
				classificationList = self.modelX[-1]

			#iterate through the possible outcomes of the class variable
			for classification in classificationList.keys():
				numeratorDict[classification] = classificationList[classification]
				#loop through outer array of the model (but we stop at second to last element of array)
				for j, attributeDict in enumerate(currModel):
					#skip the last element because this isn't an attribute -- it's the classification probabilities dictionary
					if(j == len(currModel) - 1):
						continue
					#if we run into the blank ground truth column, skip this
					if(headers[j] == dataSet.trueLabels):
						continue

					#value for the current row of the given attribute
					attrValue = row[1].iloc[j]
					if(headers[j] in dataSet.getNumericalColumns(testOrTrain)): #numerical
						meanDict = attributeDict["mean"]
						stdDict = attributeDict["std"]
						bayesNumerator = self.calculateGaussianProbability(meanDict[classification], stdDict[classification], row[1].iloc[j])
						try:
							numeratorDict[classification] += math.log(bayesNumerator)
						except:
							pass
					else: #categorical data
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
			#currently just adding dictionary of all probabilities given all classifications but eventually want to be adding the max of these (the final classification)
			for key in numeratorDict.keys():
				bayesianDict[key] = mpmath.exp(numeratorDict[key] - (max(numeratorDict.items(), key=operator.itemgetter(1))[0])) / denominatorSum

			maxClassification = max(bayesianDict.items(), key=operator.itemgetter(1))[0]
			classificationColumn.append(maxClassification)

		#sets new column equal to the array of classifications
		dataFrame["Bayes Classification"] = classificationColumn
		dataSet.resetHeaders(testOrTrain)

	'''
	Classifies the dataset and modifies the model until the discrimination score is 0
		dataSet (DataSet) - the dataset
		CHigher (str) - C+
	'''
	def modify(self, dataSet, CHigher):
		#do exactly as ModifiedBayes does except calling TwoBayes classify
		self.classify(dataSet, "train")
		dataFrame = dataSet.trainDataFrame
		protected = dataSet.protectedAttribute
		groundTruth = dataSet.trueLabels
		# need to know index of sensitive attribute in the model
		sensitiveAttributeModelIndex = dataSet.trainHeaders.index(protected)

		#Assign dictionary values based on CHigher parameter
		classesList = self.getAttributeCategories(dataFrame, dataSet.trueLabels)
		higherOrLowerClassificationDict = {}
		self.assignClassifications(higherOrLowerClassificationDict, CHigher, classesList)

		#calculate the number of people in the dataset that are actually classified as C+ (in the ground truth column - the real number from the data)
		actualNumPos = self.calculateNumPos(dataFrame, groundTruth, higherOrLowerClassificationDict)

		#Compute counts for C+S-,C-S+,C+S+,and C-S- based on counts from the original groundTruth column
		CHigherSLowerCount = self.countIntersection(dataFrame, protected, self.Sy, groundTruth, higherOrLowerClassificationDict["higher"])
		CLowerSHigherCount = self.countIntersection(dataFrame, protected, self.Sx, groundTruth, higherOrLowerClassificationDict["lower"])
		CHigherSHigherCount = self.countIntersection(dataFrame, protected, self.Sx,groundTruth, higherOrLowerClassificationDict["higher"])
		CLowerSLowerCount = self.countIntersection(dataFrame, protected, self.Sy, groundTruth, higherOrLowerClassificationDict["lower"])
		#Compute baseline probabilities based on the corresponding counts above, which will be used to calculate the preliminary disc score
		CHigherSLower = CHigherSLowerCount / self.countAttr(dataFrame, protected, self.Sy)
		CHigherSHigher = CHigherSHigherCount / self.countAttr(dataFrame, protected, self.Sx)
		CLowerSLower = CLowerSLowerCount / self.countAttr(dataFrame, protected, self.Sy)
		CLowerSHigher = CLowerSHigherCount / self.countAttr(dataFrame, protected, self.Sx)
	
		#Calculate the preliminary discrimination score -- disc = P(C+ | S+) - P(C+ | S-)
		disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)

		while (disc > 0.0):
			#Calculate numPos -- the number of instances that we classify people as C+
			numPos = self.calculateNumPos(dataFrame, "Bayes Classification", higherOrLowerClassificationDict)

			weightOfChange = 0.01 #Value by which we will be modifiying the counts

			if (numPos < actualNumPos): #We have more positive C+ labels we can assign

				#Slightly increase the count for C+S- and slightly decrease the count for C-S-
				CHigherSLowerCount = CHigherSLowerCount + (weightOfChange * CLowerSHigherCount)
				CLowerSLowerCount = CLowerSLowerCount - (weightOfChange * CLowerSHigherCount)

				#Update the probabilities based on these new counts
				CHigherSLower = CHigherSLowerCount / self.countAttr(dataFrame, protected, self.Sy)
				CLowerSLower = CLowerSLowerCount / self.countAttr(dataFrame, protected, self.Sy)

				#Overwrite the old probabilities in the model
				self.modelY[-1][higherOrLowerClassificationDict["higher"]] = CHigherSLower
				self.modelY[-1][higherOrLowerClassificationDict["lower"]] = CLowerSLower

			else: #we have assigned more positive C+ labels than we should be

				#Slightly increase the count for the C-S+ and slightly decrease the count for C+S+
				CLowerSHigherCount = CLowerSHigherCount + (weightOfChange * CHigherSLowerCount)
				CHigherSHigherCount = CHigherSHigherCount - (weightOfChange * CHigherSLowerCount)

				#Update the probabilities based on these new counts
				CLowerSHigher = CLowerSHigherCount / self.countAttr(dataFrame, protected, self.Sx)
				CHigherSHigher = CHigherSHigherCount / self.countAttr(dataFrame, protected, self.Sx)

				#Overwrite the old probabilities in the model
				self.modelX[-1][higherOrLowerClassificationDict["lower"]] = CLowerSHigher
				self.modelX[-1][higherOrLowerClassificationDict["higher"]] = CHigherSHigher

			#reclassify and recompute the new discrimination score
			self.classify(dataSet, "train")
			dataFrame = dataSet.trainDataFrame
			disc = self.calculateDiscriminationScore(CHigherSHigher, CHigherSLower)