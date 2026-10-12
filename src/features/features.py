import click
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import yaml

with open ('params.yaml','r') as f:
    params=yaml.safe_load(f)

max_features=params['build_features']['max_features']



@click.command()
@click.argument('input_filepath',type=click.Path(exists=True))
@click.argument('output_dir',type=click.Path())

def main(input_filepath,output_dir):
    logger=logging.getLogger(__name__)
    logger.info('Building Features')

    df=pd.read_csv(input_filepath)

    df["char_len"] = df["text"].apply(len)
    df["word_len"] = df["text"].apply(lambda x: len(x.split()))
    df["sentence_len"] = df["text"].apply(lambda x: x.count("."))


    X=df['text']
    y=df['label']

    le = LabelEncoder()
    y = le.fit_transform(df["label"])


    vec=TfidfVectorizer(stop_words='english',max_features=max_features)
    X_vec=vec.fit_transform(X)

    num_features=df[['char_len','word_len','sentence_len']]
    X=hstack([X_vec,num_features])

    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    joblib.dump(X_vec,output_dir/"X_vec.pkl")
    joblib.dump(y,output_dir/'y.pkl')
    joblib.dump(vec,output_dir/'tfidf_vec.pkl')
    joblib.dump(le,output_dir/'label_encoder.pkl')

    logger.info("features saved")

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main()
