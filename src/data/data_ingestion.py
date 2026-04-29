import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
# import configure
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("ERROR")

farmatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(farmatter)
file_handler.setFormatter(farmatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict: 
    try:
        with open(params_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    except FileNotFoundError:
        logger.error(f"Error: {params_path} file nahi mili.")
        raise
        
    except yaml.YAMLError as e:
        logger.error("YAML Error: %s", e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
        
def load_data(data_url: str) -> pd.DataFrame:
    try:
        # Pehle headers check karte hain
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data : {e}")
        raise

def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    try:
        # remove missing value
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug('Data preprocessing completed: Missing vales ,duplicates and empty strings row delete')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path , "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(f"Train and Test Data successfully save ho gaya: {raw_data_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while saving the data: {e}")
        raise

def main() -> None:
    try:
        # 1. Load Params
        params = load_params(params_path =os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']
 
        # 2. Read Data
        df = load_data(data_url=r"c:\Users\Pratik\OneDrive\Desktop\Youtube_sentiment_analysis\Reddit_Data.csv")

        # Preprocess the data
        final_df = preprocess_data(df)
        # 3. Split Data
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # 4. Save Data
        
        save_data(train_data, test_data,data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) , '../../data'))
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()