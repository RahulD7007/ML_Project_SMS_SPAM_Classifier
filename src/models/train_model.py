import click
import logging
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


with open ('params.yaml','r') as f:
    params=yaml.safe_load(f)

tm = params["train_model"]
model_name = tm["active_model"]
model_params = tm[model_name]

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVC,
}


@click.command()
@click.argument("features_dir", type=click.Path(exists=True))
@click.argument("model_output", type=click.Path())

def main (features_dir,model_output):
    logger=logging.getLogger(__name__)
    logger.info("training model")

    X=joblib.load(Path(features_dir)/'X_vec.pkl')
    y=joblib.load(Path(features_dir)/'y.pkl')

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**model_params)

        
    model.fit(X_train,y_train)

    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,model_output)
    logger.info('Model saved to %s',model_output)

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main()





    