import os, sys
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
# from src.components.model_evaluation import ModelEvaluation


data_ingestion = DataIngestion(
                            'data/lake/train.csv',
                            'data/lake/test.csv'
                            )
data_transformation = DataTransformation()
model_trainer_obj = ModelTrainer()

train_data_path, valid_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
df_train, df_valid, df_test = data_transformation.initialize_data_transformation(
                                                                                train_data_path, 
                                                                                valid_data_path, 
                                                                                test_data_path
                                                                                )
model_trainer_obj.initate_model_training(df_train, df_valid, df_test)

# model_eval_obj = ModelEvaluation()
# model_eval_obj.initiate_model_evaluation(train_arr,test_arr)