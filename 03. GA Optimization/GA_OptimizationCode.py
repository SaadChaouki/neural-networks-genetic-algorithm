# GA_OptimizationCode -- B115530
# ===================================================================================================================
# Code to run: GA_OptimizationCode.py
# Input: Trained ANN Model; List of ad parameters; List of users.
# Output: Performance file
# Description: Contains the genetic algorithm and its use. The file GA_Optimizer contains the object genetic algorithm
# that takes care of the optimization. GA_Functions contains different functions such as the UsersImport that import
# the users from the file and transforms them before yielding them one by one. The functions ImportCharacteristics
# imports the list of characteristics and removes the NaN values resulting in a difference of size before returning
# them. The code GA_OptimizationCode.py coordinates all the files. It imports the users, feeds them to the object Ga_
# Optimizer then optimises each user before saving the results in an excel file.
# ===================================================================================================================

from GA_Optimizer import GeneticsAlgorithm
from GA_Functions import ImportUsers, GetHeaderAdvertisements, GetHeaderUsers, ImportCharacteristics
import numpy as np
import xlsxwriter
import time
import sys

# Reconstructs the header with the users and ads parameters with the average CTR start and end.
def WriteSummaryHeader(Sheet):
    Sheet.write(0, 0, "Start - Average")
    Sheet.write(0, 1, "End - Average")
    Start = 2
    for k in GetHeaderUsers():
        Sheet.write(0, Start, k)
        Start += 1
    for k in GetHeaderAdvertisements():
        Sheet.write(0, Start, k)
        Start +=1
    Sheet.write(0, Start, "Time in seconds")
    return Sheet

# Reconstructs the header with the users and ads parameters with the user number and the CTR for each advertisement. 
def WriteHeader(Sheet):
    Sheet.write(0, 0, "User Number")
    Sheet.write(0, 1, "CTR")
    Start = 2
    for k in GetHeaderUsers():
        Sheet.write(0, Start, k)
        Start += 1
    for k in GetHeaderAdvertisements():
        Sheet.write(0, Start, k)
        Start +=1
    return Sheet

# Takes an array, sheet, and row, and write the values in the array in the excel file.
def WriteResults(Sheet, Row, Array):
    for i, value in enumerate(Array):
        Sheet.write(Row, i, value)
    return Sheet

# Genetics Parameters
CrossoverRate = 0.8
MutationRate = 0.025
PopulationSize = 70
Iterations = 75
TopAds = 10

# Files MGT
CharacteristicsPath = "01. Inputs/CharacteristicsList.csv"
UsersFilePath = "01. Inputs/Sample_Users.csv"
ModelPath = "01. Inputs/Trained_Model.h5"
TargetFilePath = "02. Ouputs/PerformanceReport.xlsx"

# Imports list characteristics
ListCharacteristics = ImportCharacteristics(CharacteristicsPath)

# Initializing the optimizer with the population size, characteristics, and model path.
Optimizer = GeneticsAlgorithm(PopulationSize=PopulationSize, PossibleValues=ListCharacteristics,PathModel=ModelPath)

# Creating the excel file
MainExcelRow = 1
SecondSheetRow = 1
ExcelFile = xlsxwriter.Workbook(TargetFilePath)
ExcelSheet = ExcelFile.add_worksheet("BestAds")
ListAdsSheet = ExcelFile.add_worksheet("Top " + str(TopAds) + " Ads")
ExcelSheet = WriteSummaryHeader(ExcelSheet)
ListAdsSheet = WriteHeader(ListAdsSheet)
CountUser = 0

# The function ImportUsers loads the users file row by row and applies the time transformation to match with the data
# the ANN was trained with. It returns the users data and the date separately as well as the original values
# to be saved.
for User, Date, Original in ImportUsers(UsersFilePath):
    StartUser = time.time()
    ArrayResults = []
    sys.stdout.write('\r' + "Optimising user: " + str(CountUser + 1))
    # Feed the user data and parameters to the optimizer. The population is generated automatically
    Optimizer.UpdateParameters(MutationRate, CrossoverRate, User, Date, PopulationSize)

    # Compute the starting fitness list after the population is generated to record the start
    Optimizer.ComputeFitnessList()

    # Record the start state
    ArrayResults.append(np.average(Optimizer.FitnessValues))

    # Optimize and get the average CTR, maximum CTR, and the best solution in the population.
    Average, Max, BestSolution = Optimizer.OptimizeSolution(Iterations)

    # Record results
    ArrayResults.append(Average)
    ArrayResults.extend(Original)
    ArrayResults.extend(BestSolution)
    ArrayResults.append(time.time()-StartUser)

    # Record in excel file and update count.
    ExcelSheet = WriteResults(ExcelSheet, MainExcelRow, ArrayResults)

    # Record the top N ads in a different sheet for each user with their CTR. The array recorded ads is used to
    # only write the unique values.
    SortedAds = sorted(zip(Optimizer.FitnessValues,Optimizer.Population))
    CountAds = 0
    RecordedAds = []
    for i in range(len(SortedAds)-1, 0, -1):
        Ad = SortedAds[i][1]
        if Ad not in RecordedAds:
            Arr = []
            Arr.append("User " + str(CountUser + 1))
            Arr.append(SortedAds[i][0])
            Arr.extend(Original)
            Arr.extend(Ad)
            WriteResults(ListAdsSheet,SecondSheetRow,Arr)
            SecondSheetRow += 1
            RecordedAds.append(Ad)
            CountAds += 1
        if CountAds == TopAds:
            break
    MainExcelRow += 1
    CountUser +=1

# Closing the excel file
ExcelFile.close()
