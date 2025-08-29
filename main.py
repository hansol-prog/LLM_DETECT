import argparse
import pickle 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

############ Scikit-Learn Implementation ############
def sklearn_experiment(train_split, ml_model):
    
    X = train_split['text']
    y = train_split['label']

    if ml_model == 'Logistic Regression':
        model = LogisticRegression(random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Training 
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, predictions)

    print(f'Validation AUC: {auc:.4f}')

    return auc

#### Main Function ####
def main():

    print("Load " + " dataset...")
    # load dataset
    with open('data/train_ml_data.pkl', 'rb') as f:
        data = pickle.load(f)
    train_dataset = data['train']

    # obtain text, feature, and label 
    train_text = []
    train_feature = []
    train_label = []

    for inst in train_dataset:
        train_text.append(inst['text'])
        train_feature.append(inst['feature'])
        train_label.append(inst['label'])

    # Prepare Training and Test Data
    train_split = {}
    train_split['text'] = train_text
    train_split['feature'] = train_feature
    train_split['label'] = train_label

    ml_models = ['Logistic Regression']
    total_experiment_results = {}
    print("Conduct Experiments for ML Models with Comma Feature with Standard Scaling..")
    for ml_model in ml_models: 
        print("Conduct Experiment for " + ml_model + "..")
        result = sklearn_experiment(train_split,ml_model)
        total_experiment_results[ml_model] = result

if __name__ == '__main__':

    main()