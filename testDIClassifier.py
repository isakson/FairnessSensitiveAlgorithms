from DataSet import DataSet
from classifierForDI import detectDI

ds = DataSet()
ds.loadData("SmallClassifiedRicciData.csv", "Race", "Class")
# ds.loadData("ClassifiedRicciData.csv", "Race", "Class")
# ds.loadData("dataCSVs/UnalteredClassifiedData/PortugueseStudent/student-por.csv", "guardian", "G1")
# ds.loadData("dataCSVs/UnalteredClassifiedData/APM_DougEvansCases.csv", "race", "struck_state")

print(ds.dataFrame)

print(detectDI(ds))