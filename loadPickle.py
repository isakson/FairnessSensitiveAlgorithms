import pickle

'''
Loads the pickle and returns the unpickled object.
    fileName (string) - the name of the pickle file to load. If "pickledObjects/" is not in fileName,
        it's automatically added.
'''
def loadPickle(fileName):
    if "pickledObjects/" not in fileName:
        fileName = "pickledObjects/" + fileName

    file = open(fileName, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj