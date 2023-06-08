import torch, os
import numpy as np
from datetime import datetime
import argparse
#from utils import _logger, set_requires_grad
#from utils import _calc_metrics, copy_Files
from Model import *

from DataLoader import data_generator
from Trainer import Trainer, model_finetune, model_test
from configs.SleepEEG_configs import Config as SleepEEG_Config
from configs.Epilepsy_configs import Config as Epilepsy_Config

"""Train and validate a model.
First, a parser is prepared to save the relevant information relating to the training run"""

start_time = datetime.now()

"""The Trainer method requires a model and a dataset at its most basic, as well as configs for the dataset."""
source_dataset = "SleepEEG" # This needs to be changed to the correct dataset
target_dataset = "Epilepsy"
configs = SleepEEG_Config()
sourcedata_path = os.path.join("datasets", source_dataset)
targetdata_path = os.path.join("datasets", target_dataset)
# This function decides which augmentations are used, as data are augmented within the TimeSeriesDataSet class
train_loader, valid_loader, test_loader = data_generator(sourcedata_path=sourcedata_path, targetdata_path=targetdata_path,
                                                         config = configs, augment=True, jitter=True, scaling=True,
                                                         addition=True)

TFC_model = TFC_Classifer(configs = configs)
classifier = target_classifier(configs).to(device = None)

temporal_contr_model = None
temporal_contr_optimizer = None

model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr = configs.lr, betas = (configs.beta1, configs.beta2), weight_decay = 3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr = configs.lr, betas = (configs.beta1, configs.beta2), weight_decay = 3e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Pre-train a model"""
Trainer(TFC_model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_loader, valid_loader, test_loader,
        device = device, logger = None, config = configs, experiment_log_dir = os.getcwd(), training_mode = "pre_train",
        classifier = classifier, classifier_optimizer = classifier_optimizer)

#Load pre-trained model:
pre_trained_model_path = os.path.join("Saved models", "ckp_last.pt")

checkpoint = torch.load(pre_trained_model_path, map_location=device)
pre_trained_dict = checkpoint["model_state_dict"]
#Load model into TFC classifier instance
TFC_model.load_state_dict(pre_trained_dict)

"""Fine-tune a model"""
Trainer(TFC_model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_loader, valid_loader,
        test_loader, device = device, logger = None, config = configs, experiment_log_dir=os.getcwd(),
        training_mode = "Fine_tune", classifier = classifier, classifier_optimizer = classifier_optimizer)

print(f"Training time was: {datetime.now() - start_time}")