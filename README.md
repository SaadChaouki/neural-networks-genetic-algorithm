# Marketing Campaigns Optimisation with Artificial Neural Networks and Genetic Algorithms

My dissertation presented in 2018 for the Master of Science in Business Analytics at the University of Edinburgh.

## Abstract

The last few years have witnessed a significant increase in the revenues generated by the online advertisement industry. In fact, it has become a crucial feature of economic life and drove large corporations to focus solely on it. Before launching a marketing campaign, advertisers aim at finding the appropriate parameters to reach the right audience and maximise their returns for a given budget. In real-time bidding, where the ads slots are sold and purchased in real-time, advertisers have to bid proportionally to the predicted outcome. In that sense, they need to wait for the adequate users to bid and potentially show their ads which might lead to a delay in results if the right users do not visit the publishers’ websites. This paper proposes a dynamic display advertisements creation methodology using a genetic algorithm and an artificial neural network to tackle this issue. Based on the user, the methodology finds the right ad parameter to set that maximise the probability of occurrence of a click. Thus, allowing the advertisers to adapt their ads to each user. The methodology is tested on a real-life dataset and shows a significant improvement in the predicted CTR for the newly constructed ads from 36% to 91% while being able to handle parameters constraints, new, and returning users. In-depth experimenting further shows that the methodology’s advantages can be extended to select the appropriate audience and select the relevant parameters to a given audience. Making the methodology a valuable tool for the entire marketing campaigns decision making.



## Original Data
Contains a sample from the original data. The original file is too large to be included in this file.

## Data Handling
* **Code to run**: Data_Selection -> SMOTE -> Data_Transformation
* **Input**: Original Data
* **Output**: SMOTEd Training, Validation, and testing files.
* **Description**: Contains the Python code for data selection and data transformation as well as the R script for SMOTE using overlap. Data_Selection selects a subset of 10% from the original sample and divides it into training, testing and validation. Data_Transformation regroups the training, validation, and testing files and transforms the values before hashing them and saving the files again. The data can be split inside the R-Script if the file takes too much memory. This is not the case with the sample used but may be needed if the data is too large to be loaded at once. This is performed by splitting the data into blocks before feeding it to SMOTE then rebuild the dataset after it was oversampled.


## ANN Training
* **Code to run**: ANN_Training.py
* **Input**: SMOTEd Training, Validation, and testing files.
* **Output**: Trained ANN model; model performance on testing and validation. 
* **Description**: Contains the Python code that trains the model. The function BuildANNModel takes as input various parameters that can be easily changed. The code starts loading the training and validation data then builds the model and compiles it. The model is then trained and the validation set is used as a validation. When the training is over, the performance on the validation set is written in an excel file in terms of AUC, weighted F-Score and accuracy. The testing file is then loaded to test the performance of the model on it. Finally, the model is saved as an .h5 file to be used for the GA optimisation.


## GA Optimization
* **Code to run**: GA_OptimizationCode.py
* **Input**: Trained ANN Model; List of ad parameters; List of users.
* **Output**: Performance file
* **Description**: Contains the genetic algorithm and its use. The file GA_Optimizer contains the object genetic algorithm that takes care of the optimization. GA_Functions contains different functions such as the UsersImport that import the users from the file and transforms them before yielding them one by one. The functions ImportCharacteristics imports the list of characteristics and removes the NaN values resulting in a difference of size before returning them. The code GA_OptimizationCode.py coordinates all the files. It imports the users, feeds them to the object Ga_Optimizer then optimises each user before saving the results in an excel file.

## Library Requirements
### Python Libraries: 
* Numpy
* Pandas
* CSV
* SYS
* OS
* Shutil
* Time
* Datetime
* Keras
* Sklearn; 
* xlsxwriter
* random
* hashlib

### R Libraries: 
* UBL
