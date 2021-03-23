# ANN Training -- B115530
# ===================================================================================================
# 03. ANN Training
# Input: ANN_Training.py
# Output: Trained ANN model; model performance on testing and validation.
# Description: Contains the Python code that trains the model. The function BuildANNModel takes as input various
# parameters that can be easily changed. The code starts loading the training and validation data then builds the model
# and compiles it. The model is then trained and the validation set is used as a validation. When the training is over,
# the performance on the validation set is written in an excel file in terms of AUC, weighted F-Score and accuracy.
# The testing file is then loaded to test the performance of the model on it. Finally, the model is saved as an .h5 file
# to be used for the GA optimisation.
# ===================================================================================================

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.utils import to_categorical
import pandas as pd
import xlsxwriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# Build ANN Model: Takes various inputs and returns a compiled ANN model ready for training. 
def BuildANNModel(InputDim, Number_Layers, Hidden_Activation, Output_Activation, Optimizer, Layers_Nodes):
    DNN = Sequential()
    DNN.add(Dense(Layers_Nodes[0], input_dim=InputDim, kernel_initializer=keras.initializers.random_uniform(),
                  activation=Hidden_Activation, name="Input_Layer"))
    for i in range(Number_Layers):
        DNN.add(Dense(Layers_Nodes[i + 1], kernel_initializer=keras.initializers.random_uniform(),
                      activation=Hidden_Activation, name=("Hidden_Layer_" + str(i+1))))
    DNN.add(Dense(2, kernel_initializer=keras.initializers.random_uniform(), activation=Output_Activation,
                  name = "Output_Layer"))
    DNN.compile(loss=keras.losses.binary_crossentropy, optimizer=Optimizer, metrics=['acc'])
    return DNN


# PARAMETERS OF THE ANN. The Layers_Nodes contains the number of nodes in each layer and has to be equal to Number
# _Layers +1 for the output of the first layers and the additional number of hidden layers.
Number_Epochs = 60
Batch_Size = 20
Number_Layers = 4
Optimizer = keras.optimizers.adam(lr=0.00001)
Hidden_Activation = "relu"
Output_Activation = "sigmoid"
Layers_Nodes = [104, 104, 104, 104, 104]

# Files MGT
ModelPath = "02. Outputs/Trained_Model.h5"
ExcelFilePath = "02. Outputs/Model_Performance.xlsx"
TrainingFile = "01. Inputs/Training.csv"
ValidationFile = "01. Inputs/Validation.csv"
TestingFile = "01. Inputs/Testing.csv"

# Importing the data. The target column (click) is changes to 2 columns representing the positive and negative
# samples for training purposes.
print("----------------------------------------------")
print("Loading Training File ...")
TRAINFILE = pd.read_csv(TrainingFile)
Train_y = to_categorical(TRAINFILE['click'].values)
del TRAINFILE['click']
Train_x = TRAINFILE.values
del TRAINFILE

print("Loading Validation Data ...")
VALFILE = pd.read_csv(ValidationFile)
Val_y = to_categorical(VALFILE['click'].values)
Original_Val_y = VALFILE['click'].values
del VALFILE['click']
Val_x = VALFILE.values
del VALFILE


# Create the excel file to save results
print("Creating Excel File ...")
ExcelFile = xlsxwriter.Workbook(ExcelFilePath)
ExcelSheet = ExcelFile.add_worksheet("Data")
ExcelSheet.write(0,0,"Configuration")
ExcelSheet.write(0,1,"Normal Proba AUC")
ExcelSheet.write(0,2,"Accuracy")
ExcelSheet.write(0,3,"Weighted F-Score")
ExcelLine = 1

# Messages
print("Length Training Data: " + str(len(Train_y)))
print("Length Validation Data: " + str(len(Val_y)))

# Building the model
Model = BuildANNModel(InputDim=Train_x.shape[1],Number_Layers=Number_Layers, Hidden_Activation=Hidden_Activation
                      , Output_Activation=Output_Activation, Optimizer=Optimizer, Layers_Nodes=Layers_Nodes)

# MODEL TRAINING
print("Start Training.")
Model.fit(Train_x, Train_y, verbose=True, batch_size=Batch_Size, shuffle=True, validation_data=(Val_x, Val_y),
          epochs=Number_Epochs)


# Model Evaluation
PredictedProbabilities = Model.predict_proba(Val_x)
PredictdClasses = Model.predict_classes(Val_x).reshape(-1, 1)
Predicted_y = []
for k in PredictdClasses:
    Predicted_y.append(k[0])

# Metrics Calculation
print("Evaluating Model Performance.")
ACCScore = accuracy_score(Original_Val_y, Predicted_y)
Weighted_FScore = f1_score(Original_Val_y, Predicted_y,average="weighted")
Proba_AUC = roc_auc_score(Val_y, PredictedProbabilities)

# Writing in the excel file
print("Saving Results.")
ExcelSheet.write(ExcelLine,0,"Validation Performance")
ExcelSheet.write(ExcelLine,1,Proba_AUC)
ExcelSheet.write(ExcelLine,2,ACCScore)
ExcelSheet.write(ExcelLine,3,Weighted_FScore)
ExcelLine+= 1

# Evaluating on testing file - Load the testing file.
print("Loading Testing Data ...")
TESTFILE = pd.read_csv(TestingFile)
Test_y = to_categorical(TESTFILE['click'].values)
Original_Test_y = TESTFILE['click'].values
del TESTFILE['click']
Test_x = TESTFILE.values
del TESTFILE

# Model Evaluation
PredictedProbabilities = Model.predict_proba(Val_x)
PredictdClasses = Model.predict_classes(Val_x).reshape(-1, 1)
Predicted_y = []
for k in PredictdClasses:
    Predicted_y.append(k[0])

# Metrics Calculation
print("Evaluating Model Performance.")
ACCScore = accuracy_score(Original_Test_y, Predicted_y)
Weighted_FScore = f1_score(Original_Test_y, Predicted_y,average="weighted")
Proba_AUC = roc_auc_score(Test_y, PredictedProbabilities)

# Writing in the excel file
print("Saving Results.")
ExcelSheet.write(ExcelLine,0,"Testing Performance")
ExcelSheet.write(ExcelLine,1,Proba_AUC)
ExcelSheet.write(ExcelLine,2,ACCScore)
ExcelSheet.write(ExcelLine,3,Weighted_FScore)

# Saving the model
Model.save(ModelPath)

# Closing the file
ExcelFile.close()
