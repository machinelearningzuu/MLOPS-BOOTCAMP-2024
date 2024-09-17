import os, sys
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.utils.utils import load_object
from src.exception.exception import customexception

class CustomData:
    def __init__(
                self,
                sample_json
                ):
        self.sample_json = sample_json
            
    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame(self.sample_json, index=[0])
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)
        
class PredictPipeline:
    def __init__(self):
        self.load_artifacts()

    def load_artifacts(self):
        scalar_obj_path=os.path.join('artifacts','scalar.pkl')
        imputer_cat_obj_path=os.path.join('artifacts','imputer_cat.pkl')
        imputer_num_obj_path=os.path.join('artifacts','imputer_num.pkl')
        label_encoder_obj_path=os.path.join('artifacts','label_encoder.pkl')
        trained_model_file_path = os.path.join('artifacts','classifier.pkl')

        self.scalar=load_object(scalar_obj_path)
        self.imputer_cat=load_object(imputer_cat_obj_path)
        self.imputer_num=load_object(imputer_num_obj_path)
        self.label_encoder=load_object(label_encoder_obj_path)
        self.model=load_object(trained_model_file_path)

    def inference_data_pipe(self,features):
        try:
            features = features.drop('Loan_ID', axis=1)
            num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

            features[num_columns] = self.imputer_num.transform(features[num_columns])
            features[cat_columns] = self.imputer_cat.transform(features[cat_columns])

            for col in cat_columns:
                features[col] = self.label_encoder[col].transform(features[col])

            features[num_columns] = np.log1p(features[num_columns])

            features[num_columns + cat_columns] = self.scalar.transform(features[num_columns + cat_columns])
            return features

        except Exception as e:
            raise customexception(e,sys)
        
    def predict(
                self,
                features,
                output_columns = ['Loan_Status']
                ):
        try:
            scaled_fea = self.inference_data_pipe(features)
            pred = self.model.predict(scaled_fea)
            pred = int(pred[0])
            label = self.label_encoder[output_columns[0]].inverse_transform([pred])
            return label[0]
        except Exception as e:
            raise customexception(e,sys)
        
# sample_json = {
#             "Loan_ID":"LP001002",
#             "Gender":"Male",
#             "Married":"No",
#             "Dependents":"0",
#             "Education":"Graduate",
#             "Self_Employed":"No",
#             "ApplicantIncome":5849,
#             "CoapplicantIncome":0.0,
#             "LoanAmount":None,
#             "Loan_Amount_Term":360.0,
#             "Credit_History":1.0,
#             "Property_Area":"Urban"
#             }

# ppipe = PredictPipeline()
# idata_obj = CustomData(sample_json)

# features = idata_obj.get_data_as_dataframe()
# pred = ppipe.predict(features)
# print(pred)