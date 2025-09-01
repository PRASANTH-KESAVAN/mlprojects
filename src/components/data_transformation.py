import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_path = os.path.join('artifact', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            
            logging.info("Numerical pipeline completed.")
            
            
            
            cat_pipeline =Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical pipeline completed")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
                
            )
            
            logging.info("Preprocessor Pipeline completed.")
            
            
            return preprocessor

            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            preprocess_obj = self.get_data_transformer_obj()
            
            logging.info("Reading the train and test datas.")
            
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            
            
            
            input_features_train_df= train_df.drop(columns=target_column_name, axis=1)
            target_train_df = train_df[target_column_name]
            
            input_features_test_df= test_df.drop(columns=target_column_name, axis=1)
            target_test_df = test_df[target_column_name]
            
            logging.info("Fitting them  to preprocessor line 85")
            
            
            input_feature_train_arr = preprocess_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocess_obj.transform(input_features_test_df)
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]
            
            logging.info("Saving preprocessor pickle file")
            
            
            
            save_object(
                self.data_tranformation_config.preprocess_path,
                preprocess_obj
                
            )
            
            logging.info("Data transformation completed.")
            
            
            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocess_path
            )
            
            
            
        except Exception as e:
            raise CustomException(e, sys)
