import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Pregnancies:int,
                 Glucose:int,
                 BloodPressure:int,
                 SkinThickness:int,
                 Insulin:int,
                 BMI:float,
                 DiabetesPedigreeFunction:float,
                 Age:int,):
        
        self.Pregnancies=Pregnancies
        self.Glucose=Glucose
        self.BloodPressure=BloodPressure
        self.SkinThickness=SkinThickness
        self.Insulin=Insulin
        self.BMI=BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Pregnancies':[self.Pregnancies],
                'Glucose':[self.Glucose],
                'BloodPressure':[self.BloodPressure],
                'SkinThickness':[self.SkinThickness],
                'Insulin':[self.Insulin],
                'BMI':[self.BMI],
                'DiabetesPedigreeFunction':[self.DiabetesPedigreeFunction],
                'Age':[self.Age],
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)