import click
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("features_dir", type=click.Path(exists=True))

def main(model_path,features_dir):
    model=joblib.load(model_path)

    X=joblib.load(Path(features_dir)/'X_vec.pkl')
    y=joblib.load(Path(features_dir)/'y.pkl')

    predictions=model.predict(X)
    acc=accuracy_score(y,predictions)

    print(f'model accuracy:{acc:4f}')

if __name__=="__main__":
    main()

    