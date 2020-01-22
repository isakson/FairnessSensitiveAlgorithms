from DataSet import DataSet
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


'''
Splits data into training and test sets by shuffling the order of the rows and splitting into 
two DataFrames according to the ratios provided.
    dataFrame (DataFrame) - the dataFrame to split into training and test sets
    fractionTrain (float between 0 and 1) - the percentage of the data that should be in the training set
    fractionTest (float between 0 and 1) - the percentage of the data that should be in the training set
        Note: fractionTrain and fractionTest should sum to 1.
    classifierCol (string) - the header for the column that the classifier should be trying to classify
'''
#TODO: write return for comment above

def splitDataIntoTrainTest(dataFrame, fractionTrain, fractionTest, classifierCol):
    train_test = train_test_split(dataFrame, train_size=fractionTrain, test_size=fractionTest, shuffle=True)

    trainClassifications = train_test[0][classifierCol]
    testClassifications = train_test[1][classifierCol]

    return train_test, trainClassifications, testClassifications

'''
Runs an SVC SVM classifier on the specified training and test sets.
'''
def classify(train_test, trainClassifications, testClassifications):
    classifier = SVC(class_weight='balanced')
    classifier.fit(train_test[0], trainClassifications)
    return classifier.predict(train_test[1])

'''
Computes the balanced error rate (BER) for the classifier results.
'''
def computeBER(train_test, classifierCol, classifierResults):
    bsr = balanced_accuracy_score(train_test[1][classifierCol], classifierResults)
    ber = 1 - bsr
    return ber

'''
Computes the parameter Beta for the algorithm.
'''
def computeBeta(train_test, classifierCol, classifierResults):
    confusionMatrix = confusion_matrix(train_test[1][classifierCol], classifierResults)
    beta = confusionMatrix[0, 1] / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
    return beta

def detectDI(dataSet):
    copy = dataSet.copyDataSet()
    copy.makeNumerical(copy.protectedAttributes[0]) #TODO: remove [0] once it's pushed onto master
    dummified = copy.dummify()

    classifierCol = copy.protectedAttributes[0] #TODO also change
    data_split = splitDataIntoTrainTest(dummified, .8, .2, classifierCol)
    classifications = classify(data_split[0], data_split[1], data_split[2])
    ber = computeBER(data_split[0], classifierCol, classifications)
    beta = computeBeta(data_split[0], classifierCol, classifications)
    epsilonPrime = 1 / 2 - beta / 8  # allowable BER threshold
    if epsilonPrime < ber:
        return "No disparate impact."
    else:
        return "Possible disparate impact."

ds = DataSet()
ds.loadData("ClassifiedRicciData.csv", ["Race"], "Class")
print(detectDI(ds))
