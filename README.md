### Overview
This project is a [Carleton College computer science](https://www.carleton.edu/computer-science/) comps (thesis/capstone project) for 2019-20. We created this directory to contain the files we need in order to research fairness-sensitive classification algorithms.

The goal of our comps was to implement several fairness-sensitive algorithms and examine 
their performance on different datasets.

For more information about what we set out to do, [click here](http://www.cs.carleton.edu/cs_comps/1920/fairness/index.php).

A website with more about our comps project and results is forthcoming.

### Running our code
All of our algorithms can be run through runExperiments.py. Experiments can be run in 
three ways: one at a time, in a numeric series (e.g. experiments 1-7), or as a list of 
specified experiments (e.g. experiments 1, 3, 5, 7).

To run a single experiment, you can call runExperiment.py on the command line with a 
single argument representing the experiment you want to run.  For example, to run experiment 7:

    python3 runExperiments.py 7
To run a numeric series of experiments, you can call runExperiment.py on the command line 
with two arguments representing the beginning and the end of the range you want to run.  
For example, to run experiments 1-4:

    python3 runExperiments.py runExperimentSeries 1 5
To run a list of experiments, you can call runExperiments.py on the command line with one 
argument representing a list of arguments to run (with no spaces between list entries). 
For example, to run experiments 1, 3, and 5:

	python3 runExperiments.py 1,3,5

The numbers refer to the experiment number according to the table in experimentalDesign.csv,
 and runExperiments.py references the associated config file for the experiment number.

### What are the configs?
The files in the config directory contain information about each of the experiments we ran 
(e.g. a path to the dataset, a filename for the output, which type of classifier to run, etc.).
 The numbers refer to the experiment number according to the table in experimentalDesign.csv.

### File organization
Experimental results (from running runExperiments.py) are written to the results directory.

All of the data we used for our experiments can be found in the originalData/EditedClassifiedData
 directory. We have cleaned and modified this data from its original form (citations are listed 
 in the Dataset citations section).

CSVs of the datasets after running the Feldman et al. (2015) algorithm are written to the 
dataCSVs/repairedData directory.

CSVs of the datasets after running a classifier are written to the dataCSVs/classifiedData 
directory.

Pickled DataSet objects (after running the Feldman et al. (2015) algorithm and after running 
a classifier) are written to pickledObjects/repairedData and pickledObjects/classifiedData, 
respectively.

### Algorithm citations
The versions of Naive Bayes, Modified Bayes, and Two Bayes that we implemented aimed to 
replicate the algorithms proposed in the following paper:
Calders, T., & Verwer, S. (2010). <em>Three naive Bayes approaches for discrimination-free 
classification</em>. Data Mining and Knowledge Discovery, 21(2), 277-292.

The versions of the Feldman Repair Algorithm and Disparate Impact Detector that we 
implemented aimed to replicate the algorithms proposed in the following paper:
Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015,
August). Certifying and removing disparate impact. In <em>Proceedings of the 21th ACM SIGKDD 
International Conference on Knowledge Discovery and Data Mining</em> (pp. 259-268). ACM.

The general structure of our project is thanks to:
Friedler, S. A., Scheidegger, C., Venkatasubramanian, S., Choudhary, S., Hamilton, E. P., 
& Roth, D. (2019, January). A comparative study of fairness-enhancing interventions in 
machine learning. In <em>Proceedings of the Conference on Fairness, Accountability, and 
Transparency</em> (pp. 329-338). ACM.

Our fairness metric implementations are based off of definitions in the following paper:
Gajane, P., & Pechenizkiy, M. (2017). On formalizing fairness in prediction with machine 
learning.

### Dataset citations
Craft, W., Montgomery, D., Tungekar, R., & Yesko, P. (2018). Jurors. American Public Media. https://github.com/APM-Reports/jury-data/blob/master/jurors.csv

Dua, D., & Graff, C. (2017). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. http://archive.ics.uci.edu/ml

Health, N. Y. C. D. of, & Hygiene, M. (2020). DOHMH New York City Restaurant Inspection Results. New York City Department of Health and Mental Hygiene. https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j

Miao, W. (2010). Did the results of promotion exams have a disparate impact on minorities? Using statistical evidence in Ricci v. DeStefano. Journal of Statistics Education, 18(3).


