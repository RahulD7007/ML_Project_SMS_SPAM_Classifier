import click
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import logging
from pathlib import Path


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("features_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())

def main (data_filepath,model_path,features_dir,output_dir):

    logger=logging.getLogger(__name__)
    logger.info("generating visualizations")

    df=pd.read_csv(data_filepath)

    model=joblib.load(model_path)

    X=joblib.load(Path(features_dir)/'X_vec.pkl')
    y=joblib.load(Path(features_dir)/'y.pkl')

    output_dir=Path(output_dir)

    output_dir.mkdir(parents=True,exist_ok=True)

    plt.figure()

    df['label'].value_counts().plot(kind='bar')
    plt.title('class distribution')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.savefig(output_dir/'class_dist.png')
    plt.close()

    logger.info('visulization saved to %s',output_dir)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()