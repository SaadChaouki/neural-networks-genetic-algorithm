# GA_Functions -- B115530
# Additional functions need during the optimisation.

import numpy
numpy.set_printoptions(suppress=True)
import pandas
import datetime
from csv import DictReader

# Returns the list of characteristics specific to the users to be saved in the excel file.
def GetHeaderUsers():
    USERSHEADER = ['hour', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                   'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']
    return USERSHEADER

# Returns the list of characteristics specific to the advertisements to be used in the excel file.
def GetHeaderAdvertisements():
    CHARACTERISTICS = ['C1', 'banner_pos']
    for i in range(14, 22):
        CHARACTERISTICS.append('C' + str(i))
    return  CHARACTERISTICS

# Imports each user independetly and applies a transformation to the feature date to match with the data the ANN was
# trained with. The function yiels the users characteristics, the new date transformed, and the original user
# to be saved.
def ImportUsers(FileName):
    for User in DictReader(open(FileName)):
        Date = ''
        List = []
        DateValues = []
        Original = []
        # Extract Data
        for Header in User:
            if Header == 'hour':
                Date = User[Header]
            else:
                List.append(User[Header])
            Original.append(User[Header])
        # Convert hour
        Hour = str(Date)[6:]
        Weekday = str(datetime.datetime(int('20' + str(Date)[:2]), int(str(Date)[2:4]), int(str(Date)[4:6])).weekday())
        for i in range(0, 7):
            if i == int(Weekday):
                DateValues.append(1)
            else:
                DateValues.append(0)
        for i in range(0, 24):
            if i == int(Hour):
                DateValues.append(1)
            else:
                DateValues.append(0)
        yield List, DateValues, Original

# Import the list of characteristics and removes the NaN values in the pandas dataframe that result in the
#  difference of size between the characteristics.
def ImportCharacteristics(FileName):
    ImportedFile = pandas.read_csv(FileName)
    ListCharacteristics = []
    for header in ImportedFile:
        UniqueValues = ImportedFile[header].unique()
        if str(UniqueValues[len(UniqueValues)-1]) == 'nan':
            UniqueValues = UniqueValues[:len(UniqueValues) - 1]
        ListCharacteristics.append(UniqueValues)
    return ListCharacteristics

