import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging 
import string 
import re 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# import configure
logger = logging.getLogger('data_preprocessing')
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

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_comment(comment):
    try:
        comment = comment.lower()

        comment = comment.strip()

        comment = re.sub(r'\n', ' ', comment)

        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '',comment)

        stop_words = set(stopwords.words('english')) - {'not','but','no','yet','however'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        logger.error(f"Error in Preprocessing comment : {e}")
        return comment


   
def normalize_text(df):
    "Appling preprocessing to the text data  in dataframe"
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization complited')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization : {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_data_path = os.path.join(data_path, "interim")
        os.makedirs(interim_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        logger.debug(f"Train and Test Data successfully save ho gaya: {interim_data_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while saving the data: {e}")
        raise

def main() -> None:
    try:
       logger.debug('starting data PreProcessing ...')
       #Fetch data from data/raw
       train_data = pd.read_csv('./data/raw/train.csv')
       test_data = pd.read_csv('./data/raw/test.csv')
   
       logger.debug('Data loaded successfully')

       train_preprocessed_data = normalize_text(train_data)
       test_preprocessed_data = normalize_text(test_data)
        # 4. Save Data
       save_data(train_data, test_data,data_path = './data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()

