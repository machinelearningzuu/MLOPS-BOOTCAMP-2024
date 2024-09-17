import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import customexception
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    valid_data_path:str=os.path.join("data","valid.csv")
    train_data_path:str=os.path.join("data","train.csv")
    test_data_path:str=os.path.join("data","test.csv")

class DataIngestion:
    def __init__(
                self,
                raw_train_path:str,
                raw_test_path:str
                ):
        self.ingestion_config=DataIngestionConfig()
        self.raw_train_path=raw_train_path
        self.raw_test_path=raw_test_path
        

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            train_data=pd.read_csv(self.raw_train_path)
            logging.info("  reading the train file from data lake")

            test_data=pd.read_csv(self.raw_test_path)
            logging.info("  reading the test file from data lake")
            
            logging.info("here i have performed train test split")
            
            train_data,valid_data=train_test_split(train_data,test_size=0.22)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            valid_data.to_csv(self.ingestion_config.valid_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.valid_data_path,
                self.ingestion_config.test_data_path
                )

        except Exception as e:
            logging.info()
            raise customexception(e,sys)