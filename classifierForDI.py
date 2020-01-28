from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

'''
Runs an sklearn SVC SVM classifier on the specified data.
    data (DataFrame) - The DataFrame object whose data the classifier should use
    classifications (column of DataFrame) - The true classifications for the data
'''
def classify(data, classifications):
    classifier = SVC(class_weight='balanced')
    classifier.fit(data, classifications)
    return classifier.predict(data)

'''
Computes the balanced error rate (BER) for the classifier results.
    data (DataFrame) - The DataFrame object to compute BER for
    classifierCol (string) - The column header for the true classifications (protected attribute) in the data
    classifierResults (array) - An array of classifications (guessed by the classifier)
'''
def computeBER(data, classifierCol, classifierResults):
    bsr = balanced_accuracy_score(data[classifierCol], classifierResults)
    ber = 1 - bsr
    return ber

'''
Computes the parameter Beta for the algorithm.
    data (DataFrame) - The DataFrame object to compute BER for
    classifierCol (string) - The column header for the true classifications (protected attribute) in the data
    classifierResults (array) - An array of classifications (guessed by the classifier)
'''
def computeBeta(data, classifierCol, classifierResults):
    confusionMatrix = confusion_matrix(data[classifierCol], classifierResults)
    beta = confusionMatrix[1, 1] / (confusionMatrix[1, 0] + confusionMatrix[1, 1])
    return beta

'''
Runs the full Feldman disparate impact detection algorithm. 
    dataSet (DataSet) - The DataSet object to retrieve the data from
'''
def detectDI(dataSet):
    copy = dataSet.copyDataSet()
    copy.makeNumerical(copy.protectedAttribute)
    dummifiedData = copy.dummify()

    classifierCol = copy.protectedAttribute
    classifications = classify(dummifiedData, dummifiedData[classifierCol])
    ber = computeBER(dummifiedData, classifierCol, classifications)
    beta = computeBeta(dummifiedData, classifierCol, classifications)
    epsilonPrime = 1 / 2 - beta / 8  # allowable BER threshold
    if epsilonPrime < ber:
        return "No disparate impact."
    else:
        return "Possible disparate impact."