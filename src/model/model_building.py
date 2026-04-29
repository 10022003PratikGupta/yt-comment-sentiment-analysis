import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging 
import string 
import re 
import nltk 
import pickle
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
logger = logging.getLogger('model_building')
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
    
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NANs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data : {e}")
        raise

def train_lgbm(x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int):
    try:
        best_model = lgb.LGBMClassifier(objective='multiclass',num_class=3,metric="multi_logloss",is_unbalance=True,class_weight="balanced",reg_alpha=0.1,reg_lamba=0.1,learning_rate=learning_rate,max_depth=max_depth,n_estimators=n_estimators)
        best_model.fit(x_train, y_train)
        logger.debug('lightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error(f"Error during LigtfGBM model training : %s : {e}")
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        x_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        x_train_tfidf = vectorizer.fit_transform(x_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {x_train_tfidf.shape}")

        #Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(),'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug("TF-IDF applied with trigrams and data transformed")
        return x_train_tfidf, y_train
    except Exception as e:
        logger.error("Error during TF-IDf transformation: %s", e)
        raise     

def save_model(model, file_path: str)-> None:
    "Save trained model to a file"
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model : %s', e)
        raise        

def get_root_directory()-> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(params_path =os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_ranges'])

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        train_data = load_data(os.path.join(root_dir,'data/interim/train_processed.csv'))
        x_train_tfidf , y_train = apply_tfidf(train_data,max_features, ngram_range)

        best_model = train_lgbm(x_train_tfidf,y_train,learning_rate,max_depth,n_estimators)

        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")    


if __name__ == "__main__":
    main()
