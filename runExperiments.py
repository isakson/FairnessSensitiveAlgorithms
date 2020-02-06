import json
from pipeline import pipeline

'''
Parse a config file.
Parameters:
    num (int) - The number of the experiment (corresponds to the number of the config file)
Returns 6 stings representing the config's datasetPath, filename, protectedAttribute, 
    groundTruth, feldman, and bayes values.
'''
def parseConfig(num):
    with open("configs/" + str(num) + ".json") as f:
        config = json.load(f)
    print(config["num"])
    return config["filePath"], config["filename"], config["protectedAttribute"],\
           config["groundTruth"], config["feldman"], config["bayes"]

'''
Run a single experiment by its number.
Parameters:
    num (int) - The number of the experiment
'''
def runExperiment(num):
    config = parseConfig(num)
    pipeline(config[0], config[1], config[2], config[3], config[4], config[5])

'''
Run multiple experiments by their numbers.
Parameters:
    start (int) - The first experiment to run
    end (int) - The last experiment to run (NOT inclusive)
'''
def runExperimentSeries(start, end):
    for i in range(start, end):
        config = parseConfig(i)
        pipeline(config[0], config[1], config[2], config[3], config[4], config[5])

'''
Run a specified list of experiments.
Parameters:
    list (list of ints) - A list of experiments to run.
'''
def runExperiments(list):
    for num in list:
        config = parseConfig(num)
        pipeline(config[0], config[1], config[2], config[3], config[4], config[5])

# TODO: add command-line arguments