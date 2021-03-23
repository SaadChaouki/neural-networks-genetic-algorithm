# Set the working directory as current working directory.
# Installing and loading library
install.packages("UBL")
library(UBL)

# Recording the source file
DataPath = "02. Ouputs/02. Training File/Training.csv"

# Importing the data
Data = read.csv(DataPath,colClasses="factor")
RepartionClicks = table(Data$click)

# Applying SMOTE
Data = SmoteClassif(click ~ ., Data, C.perc = list("0" = 1,"1" = RepartionClicks[1]/RepartionClicks[2]), dist = "Overlap")

# Shuffle
Data <- Data[sample(nrow(Data)),]

# Saving the file
write.csv(Data, file = DataPath, row.names = FALSE)

