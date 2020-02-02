import json
from pipeline import pipeline

def readConfig(num):
    input_string = open("configs/" + str(num) + ".json", 'r').read()
    settings = json.loads(input_string)
    return settings["dataset"], settings["filename"], settings["protectedAttribute"], \
           settings["groundTruth"], settings["feldman"], settings["bayes"]


for i in range(1, 3):
    config = readConfig(i)
    print("Config for " + str(config[0]) + " data with filename " + str(config[1]))
    # pipeline(config)

config = readConfig(1)
pipeline(config[0], config[1], config[2], config[3], config[4], config[5])