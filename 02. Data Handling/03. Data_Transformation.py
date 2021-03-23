# Data_Transformation.py -- B115530
# ===================================================================================================
# Code to run: Data_Selection -> SMOTE -> Data_Transformation
# Input: Original Data
# Output: SMOTEd Training, Validation, and testing files.
# Description: Contains the Python code for data selection and data transformation as well as the R script for
# SMOTE using overlap. Data_Selection selects a subset of 10% from the original sample and divides it into training,
# testing and validation. Data_Transformation regroups the training, validation, and testing files and transforms the
# values before hashing them and saving the files again. The data can be split inside the R-Script if the file takes
# too much memory. This is not the case with the sample used but may be needed if the data is too large to be loaded
# at once. This is performed by splitting the data into blocks before feeding it to SMOTE then rebuild the dataset
# after it was oversampled.
# ===================================================================================================


from csv import DictReader
import pandas, numpy, csv, sys
import time
import datetime
import hashlib


# Hash function: Hashes the values using SHA1 and applies a modulo 10^6.
def DataHasher(Value):
    return int(hashlib.sha1(str(Value).encode('utf-8')).hexdigest(), 16) % 10**6

# Get the headers: Get the header from the original dataset to be used when saving the subset.
def GetHeader(FilePath):
    OriginalData = pandas.read_csv(FilePath, nrows=1)
    Array = []
    for k in OriginalData.keys():
        Array.append(k)
    return Array

# Sample Data Generator: Because the original file is too large to be loaded into memory, this function reads the data
# line by line and, based on a probability, select random rows to be saved.
def SampleDataGenerator(FilePath, Probability):
    for Sample in DictReader(open(FilePath)):
        if numpy.random.uniform(0,1) <= Probability:
            Array = []
            for Header in Sample:
                Array.append(Sample[Header])
            yield Array

# Create Sample Data: creates a csv file containing a subset of the original data.
def CreateSampleData(ORIGINALPATH, SAMPLEPATH, PROBABILITIY):
    # Open sample file and write header
    SampleFile = open(SAMPLEPATH, "w", newline='')
    csv.writer(SampleFile, lineterminator='\n', delimiter=',').writerow(GetHeader(ORIGINALPATH))
    # Save the lines in the new sample file.
    for i, Line in enumerate(SampleDataGenerator(ORIGINALPATH, Probability=PROBABILITIY)):
        csv.writer(SampleFile, lineterminator='\n', delimiter=',').writerow(Line)
        sys.stdout.write('\r' + "Saved Lines: " + str(i))
    print("\n")


if __name__ == "__main__":

    OriginalFilePath = "01. Inputs/Sample_OriginalData.csv"
    SampleFilePath = "02. Ouputs/01. Sample File/Sample.csv"
    TrainingFilePath = "02. Ouputs/02. Training File/Training.csv"
    TestingFilePath = "02. Ouputs/03. Testing File/Testing.csv"
    ValidationFilePath = "02. Ouputs/04. Validation File/Validation.csv"
    OriginalTraining = "02. Ouputs/00. Original Files/Training_Raw.csv"
    OriginalTesting = "02. Ouputs/00. Original Files/Testing_Raw.csv"
    OriginalValidation = "02. Ouputs/00. Original Files/Validation_Raw.csv"
    ProbabilitySelection = 0.1


    # Loading the data and putting the files together with keys to resaparate later. 
    print("Starting Data Transformation.")
    time.sleep(2)
    print("Loading data for transformation...")
    TrainFile = pandas.read_csv(TrainingFilePath)
    TestFile = pandas.read_csv(TestingFilePath)
    ValidationFile = pandas.read_csv(ValidationFilePath)
    DataFile = pandas.concat([TrainFile, TestFile, ValidationFile], keys=[0, 1, 2])
    print("Length of data: " + str(len(DataFile)))

    print("Data Transformation ...")
    # Dictonaries
    HoursDict = {}
    DayDict = {}
    WeekDay = {}

    # Creating Dictionarries Dates
    print("Creating Dictionnaries ...")
    for UniqueDate in DataFile["hour"].unique():
        HoursDict[UniqueDate] = str(UniqueDate)[len(str(UniqueDate)) - 2:]
        WeekDay[UniqueDate] = str(datetime.datetime(2014, 10, int(
            str(UniqueDate)[len(str(UniqueDate)) - 4:len(str(UniqueDate)) - 2])).weekday())

    # Duplicate
    DataFile['weekday'] = DataFile['hour']

    # Data Replacement
    print("Replacing Data ...")
    DataFile["hour"] = DataFile["hour"].replace(HoursDict)
    DataFile["weekday"] = DataFile["weekday"].replace(WeekDay)

    # Convert weekday to dummies
    # Training
    Dummies = pandas.get_dummies(DataFile["weekday"])
    DataFile = DataFile.drop('weekday', axis=1)
    for i, k in enumerate(Dummies):
        DataFile["day_" + str(i)] = Dummies[k]

    # Convert Hours to dummies
    Dummies = pandas.get_dummies(DataFile["hour"])
    DataFile = DataFile.drop('hour', axis=1)
    for i, k in enumerate(Dummies):
        DataFile["hour_" + str(i)] = Dummies[k]

    # Hash Control: The columns not to hash
    NoHashColumns = ['click']
    for i in range(30):
        NoHashColumns.append('day_' + str(i))
        NoHashColumns.append('hour_' + str(i))

    # The columns to delete
    ColumnsToDelete = ['id']

    # Transofmration: applying the hash, deleting the columns that have to be deleted.
    for ColumnName in DataFile.keys():
        if ColumnName in ColumnsToDelete:
            del DataFile[ColumnName]
        elif ColumnName not in NoHashColumns:
            DataFile[ColumnName] = DataFile[ColumnName].apply(DataHasher)
        else:
            continue

    # Saving: Saving the last files.
    print("Saving the transformed files ...")
    DataFile.xs(0).to_csv(TrainingFilePath, sep=",", index=False)
    DataFile.xs(1).to_csv(TestingFilePath, sep=",", index=False)
    DataFile.xs(2).to_csv(ValidationFilePath, sep=",", index=False)
