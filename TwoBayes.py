import NaiveBayes
import Bayes

class TwoBayes(NaiveBayes):
	
	def __init__(self):
		self.modelX = []
		self.modelY = []
		self.Sx = ""
		self.Sy = ""

	'''Assigns the keys "higher" and "lower" to the two possible sensitive attribute values based on which of the two has a higher count.
   S+ ("higher") is the privileged group. We do this based on counts instead of as a manual parameter because there isn't an 'ideal' 
   sensitive attribute category like there is with classifications.'''
	def assignSensitivity(self, dataSet):
		dataFrame = dataSet.dataFrame
		sensitiveAttrCatList = self.getAttributeCategories(dataFrame, dataSet.protectedAttribute)
		Sa = dataFrame.loc[dataFrame[dataSet.protectedAttribute] == sensitiveAttrCatList[0], dataSet.protectedAttribute].count()
		Sb = dataFrame.loc[dataFrame[dataSet.protectedAttribute] == sensitiveAttrCatList[1], dataSet.protectedAttribute].count()
		if (Sa > Sb):
			self.Sx = sensitiveAttrCatList[0]
			self.Sy = sensitiveAttrCatList[1]
		else:
			self.Sx = sensitiveAttrCatList[1]
			self.Sy = sensitiveAttrCatList[0]

	def trainTwo(self, dataSet):
		self.assignSensitivity(dataSet)
		#divide dataSet into two dataSets and call them dataSetX and dataSetY respectively
		
		self.train(dataSetX, self.modelX)
		self.train(dataSetY, self.modelY)
		
	def classify(self, dataSet):
		#do same as naive classify, except with an if-statement for each row to use either X or Y
		
	def modify(self, dataSet, CHigher):
		#do exactly as ModifiedBayes does except calling TwoBayes classify
		
	