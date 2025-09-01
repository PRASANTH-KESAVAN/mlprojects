
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifact", "train.csv")
    test_path: str = os.path.join("artifact", "test.csv")
    raw_path: str = os.path.join("artifact", "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            
            logging.info("Read the datset")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info("Splitting the dataset to train and test.")
            
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)
            
            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            
            logging.info("Data Ingestion is completed.")
            
            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )         
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__== "__main__":
    obj = DataIngestion()
    train_path, test_path= obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    