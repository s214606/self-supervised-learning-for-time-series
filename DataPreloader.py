## Import libraries
import torch, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

import torch.fft as fft




#load dataset
class WISDMDataset():
    def __init__(self, sensorDevice:str = 'phone', sensor:str = 'accel', testSize = 0.2, valSize = 0.2, windowSize = 128, 
                 trainIdx = None, testIdx = None, valIdx = None):
        # #Local config
        self.seed = 0
        np.random.seed(self.seed)
        self.testSize = testSize
        self.trainSize = 1 - testSize - valSize
        self.valSize = valSize/(1 - testSize)
        self.trainIdx = trainIdx
        self.testIdx = testIdx
        self.valIdx = valIdx
        
        
        
        self.windowSize = windowSize
        self.sensorDevice = sensorDevice #is phone or watch
        self.sensor = sensor #is accel or gyro
        self.basepath = "datasets\\wisdm-dataset\\raw"
        
        self.datapath = os.path.join(self.basepath,self.sensorDevice,self.sensor) #where the data is
        self.filenames = os.listdir(path=self.datapath)
        #Remove .DS_Store from list if its there
        if '.DS_Store' in self.filenames:
            self.filenames.remove('.DS_Store')
        
        self.NSamples = len(self.filenames)
        
        labelEncoder = LabelEncoder()
        
        #Split NSamples into train and test lists of integers between 0 and NSamples, with train test split from sklearn
        if type(self.trainIdx) == type(None) and type(self.testIdx) == type(None) and type(self.valIdx) == type(None):
            self.trainIdx, self.testIdx = train_test_split(np.arange(self.NSamples), test_size = self.testSize, random_state = self.seed, shuffle = True)
            self.trainIdx, self.valIdx = train_test_split(self.trainIdx, test_size = self.valSize, random_state = self.seed, shuffle = True)
        
        
        
        self.X = {iSet: np.zeros((0, self.windowSize,4)) for iSet in ["train", "test", "val"]}
        self.Y = {iSet: np.zeros((0, self.windowSize,1)) for iSet in ["train", "test", "val"]}
        self.X_aug = None
        self.X_f_aug = None
        self.y_aug = None
        #The train data set is created
                
        for nameSet, iSet in zip(["train", "test", "val"], [self.trainIdx, self.testIdx, self.valIdx]):
            print(f"Fetching data for set: {nameSet}")
            for iPerson in tqdm(iSet, total = len(iSet)):
                path = os.path.join(self.datapath,self.filenames[iPerson])
                # print(path)
                
                df = pd.read_csv(path,
                         header=None, names=["id","activity","time","x","y","z"])
                df['z'] = df['z'].str.replace(r';', '')
            
                labels = np.array(df["activity"])
                
                df['label'] = labelEncoder.fit_transform(labels)
                
                #Reset
                iTimeStep = 0 
                iActivity = None
                iData = np.zeros((1, self.windowSize,4))
                iLabels = np.zeros((1, self.windowSize,1))
                
            
                #loop over the rows in the dataframe in a for loop
                for idx, iRow in tqdm(df.iterrows(), total = len(df)):
                    if iActivity == None:
                        iActivity = iRow['label']
                    #get time, x, y, z  from the irow and put them in a numpy array
                    
                    if iActivity != iRow['label']:
                        # print(f"At timestep {iTimeStep} row {idx}, the activity changed from {iActivity} to {iRow['label']}")
                        iActivity = iRow['label']
                        iTimeStep = 0
                        iData = np.zeros((1, windowSize,4))
                        iLabels = np.zeros((1, windowSize,1))
                        
                    if iTimeStep < windowSize:
                        row = np.array([iRow['time'],iRow['x'],iRow['y'],iRow['z']])
                        iData[0, iTimeStep, :] = row
                        iLabels[0, iTimeStep, :] = iRow['label']
                        iTimeStep += 1
                        
                        #print(iData[0, iTimeStep, :], row)
                    else:
                        #iTimeStep = 0
                        #check if self.X exists, if not create it and put iData in it
                        
                        #If there are more than 1 nunique values in Ilabels, then we have a problem
                        if np.unique(iLabels).shape[0] > 1:
                            raise Exception("There are more than 1 unique values in iLabels")

                        self.X[nameSet] = np.append(self.X[nameSet], iData, axis = 0)
                        self.Y[nameSet] = np.append(self.Y[nameSet], iLabels, axis = 0)
                        iTimeStep = 0
                        iData = np.zeros((1, self.windowSize,4))
                        iLabels = np.zeros((1, self.windowSize,1))
                            
                    
        #X and Y numpy arrays are converted to tensors
        self.X["train"] = torch.from_numpy(self.X["train"]).float()
        self.X["test"] = torch.from_numpy(self.X["test"]).float()
        self.X["val"] = torch.from_numpy(self.X["val"]).float()
        self.Y["train"] = torch.from_numpy(self.Y["train"]).int()
        self.Y["test"] = torch.from_numpy(self.Y["test"]).int()
        self.Y["val"] = torch.from_numpy(self.Y["val"]).int()

    def save_data(self) -> None:
        #Save the data to a path
        if os.path.exists("datasets\\wisdm-dataset_processed") == False:
            os.mkdir("datasets\\wisdm-dataset_processed")
            
        if os.path.exists(f"datasets\\wisdm-dataset_processed\\{self.sensorDevice}{self.sensor}") == False:
            os.mkdir(f"datasets\\wisdm-dataset_processed\\{self.sensorDevice}{self.sensor}")
            
        for nameSet, iSet in zip(["train", "test", "val"], [self.trainIdx, self.testIdx, self.valIdx]):
            person_map = {idx: self.filenames[idx] for idx in iSet}

            #Save X data
            with open(f"datasets\\wisdm-dataset_processed\\{self.sensorDevice}{self.sensor}\\X_{nameSet}.pt", "wb+") as f:
                torch.save(self.X[nameSet], f)
                
            #Save activity labels
            with open(f"datasets\\wisdm-dataset_processed\\{self.sensorDevice}{self.sensor}\\Y_{nameSet}.pt", "wb+") as f:
                torch.save(self.Y[nameSet], f)
                
            #Save person indexes
            with open(f"datasets\\wisdm-dataset_processed\\{self.sensorDevice}{self.sensor}\\{nameSet}Idx.pt", "wb+") as f:
                torch.save(person_map, f)

        
        
if __name__ == "__main__":
    dataset = WISDMDataset()
    dataset.save_data()
    