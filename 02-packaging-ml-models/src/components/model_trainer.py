import pandas as pd
import os, sys, pprint
from sklearn.svm import SVC
from dataclasses import dataclass
from xgboost import XGBClassifier
from src.logger.logging import logging
from src.utils.utils import save_object
from sklearn.tree import DecisionTreeClassifier
from src.exception.exception import customexception
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','classifier.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def train_and_evaluate(
                            self, 
                            X_train,
                            Y_train,
                            X_valid,
                            Y_valid,
                            models
                            ):
        try:
            report = {}
            for i in range(len(models)):
                report[list(models.keys())[i]] = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train,Y_train)
                
                P_valid =model.predict(X_valid)
                acc_score = accuracy_score(Y_valid,P_valid)
                precision = precision_score(Y_valid,P_valid)
                recall = recall_score(Y_valid,P_valid)
                f1 = f1_score(Y_valid,P_valid)
                roc_auc = roc_auc_score(Y_valid,P_valid)

                report[list(models.keys())[i]]['accuracy'] = acc_score
                report[list(models.keys())[i]]['precision'] = precision
                report[list(models.keys())[i]]['recall'] = recall
                report[list(models.keys())[i]]['f1'] = f1
                report[list(models.keys())[i]]['roc_auc'] = roc_auc

            # convert report to dataframe
            df_report = pd.DataFrame(report)
            # make the model names as index column and metrics as columns
            df_report = df_report.T
            return df_report

        except Exception as e:
            logging.info('Exception occured during model training')
            raise customexception(e,sys)
    
    def initate_model_training(
                            self,
                            df_train, 
                            df_valid, 
                            df_test
                            ):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train = df_train.drop('Loan_Status', axis=1).values
            Y_train = df_train['Loan_Status'].values

            X_valid = df_valid.drop('Loan_Status', axis=1).values
            Y_valid = df_valid['Loan_Status'].values

            X_test = df_test.values


            models={
                    'RandomForestClassifier':RandomForestClassifier(),
                    'DecisionTreeClassifier':DecisionTreeClassifier(),
                    'XGBClassifier':XGBClassifier(),
                    'SVM':SVC()
                    }
            
            model_report = self.train_and_evaluate(X_train,Y_train,X_valid,Y_valid,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = model_report['f1'].max()
            best_model_name = model_report['f1'].idxmax()
            best_model = models[best_model_name]

            print(f'Best Model Found \nModel Name : {best_model_name} , F1 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found \nModel Name : {best_model_name} , F1 Score : {best_model_score}')

            save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                        )
            logging.info('Model Training Completed')      

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

        
    