import json
import sys
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
           config["groundTruth"], config["feldman"], config["bayes"], config["dataset"]

'''
Run a single experiment by its number.
Parameters:
    num (int) - The number of the experiment
'''
def runExperiment(num):
    config = parseConfig(num)
    pipeline(config[0], config[1], config[2], config[3], config[4], config[5], config[6])

'''
Run multiple experiments by their numbers.
Parameters:
    start (int) - The first experiment to run
    end (int) - The last experiment to run (NOT inclusive)
'''
def runExperimentSeries(start, end):
    for i in range(int(start), int(end)):
        config = parseConfig(i)
        pipeline(config[0], config[1], config[2], config[3], config[4], config[5], config[6])

'''
Run a specified list of experiments.
Parameters:
    list (list of ints) - A list of experiments to run.
'''
def runExperiments(list):
    for num in list:
        config = parseConfig(num)
        pipeline(config[0], config[1], config[2], config[3], config[4], config[5], config[6])

'''
Command line arguments:
    You can call runExperimentSeries and runExperiment from the command line.
    To call runExperimentSeries:
        Enter the argument runExperimentSeries followed by the start and end of the series you would like to run.
        For example, to run experiments 1-4:
            python3 runExperiments.py runExperimentSeries 1 5
    To call runExperiment:
        Enter a single integer parameter after the call to runExperiments.py
        For example, to run experiment 7:
            python3 runExperiments.py 7
    To call runExperiments(list):
        Enter a list as a parameter after the call to runExperiments.py (without spaces between list entries).
        For example, to run experiments 1, 3, and 5:
            python3 runExperiments.py 1,3,5 
'''
if len(sys.argv) > 1:
    if sys.argv[1] == "runExperimentSeries":
        runExperimentSeries(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "runExperiment":
        runExperiment(sys.argv[1])
    else:
        runExperiments(sys.argv[1].split(','))
