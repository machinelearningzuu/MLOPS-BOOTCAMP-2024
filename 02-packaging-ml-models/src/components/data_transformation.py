import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.utils import *
from dataclasses import dataclass
from collections import defaultdict
from src.logger.logging import logging
from sklearn.impute import SimpleImputer
from src.exception.exception import customexception
from sklearn.preprocessing import LabelEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    scalar_obj_path=os.path.join('artifacts','scalar.pkl')
    imputer_cat_obj_path=os.path.join('artifacts','imputer_cat.pkl')
    imputer_num_obj_path=os.path.join('artifacts','imputer_num.pkl')
    label_encoder_obj_path=os.path.join('artifacts','label_encoder.pkl')
    
    preprocessed_train_data_path=os.path.join('data','processed','train.csv')
    preprocessed_valid_data_path=os.path.join('data','processed','valid.csv')
    preprocessed_test_data_path=os.path.join('data','processed','test.csv')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initialize_data_transformation(
                                    self,
                                    train_path,
                                    valid_path, 
                                    test_path
                                    ):  
        try:
            num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
            output_columns = ['Loan_Status']

            df_train = pd.read_csv(train_path)
            df_valid = pd.read_csv(valid_path)
            df_test = pd.read_csv(test_path)

            logging.info(' Data Transformation initiated')

            df_train.drop('Loan_ID', axis=1, inplace=True)
            df_valid.drop('Loan_ID', axis=1, inplace=True)
            df_test.drop('Loan_ID', axis=1, inplace=True)

            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_imputer = SimpleImputer(strategy='mean')

            logging.info(' Imputer initiated')
            cat_imputer.fit(df_train[cat_columns])
            num_imputer.fit(df_train[num_columns])

            df_train[num_columns] = num_imputer.transform(df_train[num_columns])
            df_train[cat_columns] = cat_imputer.transform(df_train[cat_columns])

            df_valid[num_columns] = num_imputer.transform(df_valid[num_columns])
            df_valid[cat_columns] = cat_imputer.transform(df_valid[cat_columns])

            df_test[num_columns] = num_imputer.transform(df_test[num_columns])
            df_test[cat_columns] = cat_imputer.transform(df_test[cat_columns])

            logging.info(' Encoding initiated')
            # dictionary to store the label encoder object
            label_encoders = defaultdict(LabelEncoder)

            for col in cat_columns + output_columns:
                label_encoders[col].fit(df_train[col])

                df_train[col] = label_encoders[col].transform(df_train[col])
                df_valid[col] = label_encoders[col].transform(df_valid[col])

                if col not in output_columns:
                    df_test[col] = label_encoders[col].transform(df_test[col])

            # apply log transformation on numerical columns
            df_train[num_columns] = np.log1p(df_train[num_columns])
            df_valid[num_columns] = np.log1p(df_valid[num_columns])
            df_test[num_columns] = np.log1p(df_test[num_columns])

            logging.info(' Scaling initiated')
            scalar = StandardScaler()
            scalar.fit(df_train[num_columns + cat_columns])

            df_train[num_columns + cat_columns] = scalar.transform(df_train[num_columns + cat_columns])
            df_valid[num_columns + cat_columns] = scalar.transform(df_valid[num_columns + cat_columns])
            df_test[num_columns + cat_columns] = scalar.transform(df_test[num_columns + cat_columns])

            df_train.to_csv(self.data_transformation_config.preprocessed_train_data_path, index=False)
            df_valid.to_csv(self.data_transformation_config.preprocessed_valid_data_path, index=False)
            df_test.to_csv(self.data_transformation_config.preprocessed_test_data_path, index=False)

            save_object(self.data_transformation_config.label_encoder_obj_path, label_encoders)
            save_object(self.data_transformation_config.imputer_cat_obj_path, cat_imputer)
            save_object(self.data_transformation_config.imputer_num_obj_path, num_imputer)
            save_object(self.data_transformation_config.scalar_obj_path, scalar)

            logging.info('Data Transformation completed')

            return df_train, df_valid, df_test

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)