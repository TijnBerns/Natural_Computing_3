import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def load_data(fname):
    """Loads the data and splits it into train and test sets

    Args:
        fname: Name corersponding to datafile

    Returns:
        Training and test sets
    """
    # Read csv file
    data = pd.read_csv(fname,  header=None).sample(frac=1, random_state=0)
    
    # Transform labels into integers
    enc = LabelEncoder()
    enc.fit(data[1])
    data[1] = enc.transform(data[1])
    
    # Split into train and test set and return
    data = np.array(data)
    train, test = np.split(data, [int(0.8*len(data))])
    return (train[:, 2:].astype('float32'), train[:, 1].astype('int'), 
            test[:, 2:].astype('float32'), test[:, 1].astype('int'))


def evaluate_pred(y_true, y_pred):
    """Evaluates the prediction of a model

    Args:
        y_true: The true labels
        y_pred: The predicted labels

    Returns:
        Dictionary with different evaluation metrics for the given labels
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    return metric_dict


def ensamble(X_train, y_train, X_test, n_trees, max_depth, max_leaf_nodes):
    """Computes the prediction of an ensamble of decision trees 

    Args:
        X_train: Training dataset
        y_train: Labels corresponding to the training dataset
        X_test: Test dataset
        n_trees: Number of trees in ensamble 
        max_depth: Max depth of a single tree
        max_leaf_nodes: Maximum number of leafs of a single tree

    Returns:
        Prediction of ensamble
    """
    y_pred = np.zeros(len(X_test))
    X_train_split = np.array_split(X_train, n_trees)
    y_train_split = np.array_split(y_train, n_trees)

    for (X, y) in zip(X_train_split, y_train_split):
        clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        clf.fit(X, y)
        y_pred += clf.predict(X_test)
    return np.round(y_pred / n_trees)


def ex5(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes):
    res = []
    for n in range(1, 300):
        y_pred = ensamble(X_train, y_train, X_test, n, max_depth, max_leaf_nodes)
        res.append(evaluate_pred(y_test, y_pred)['accuracy'])
        
    plt.figure()
    plt.title(f"Max depth {max_depth}, Max leaf nodes {max_leaf_nodes}")
    plt.plot(res)
    plt.xlabel("Number of trees in ensamble")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    np.random.seed(2022)
    fname = "data/wdbc.csv"
    X_train, y_train, X_test, y_test = load_data(fname)
    depth = 1
    leafs = 2

    # Baseline
    y_pred = ensamble(X_train, y_train, X_test, 1, depth, leafs)
    accuracy = evaluate_pred(y_test, y_pred)['accuracy']
    print(f"Accuracy single tree:\t {accuracy}")

    # Ensamble 
    y_pred = ensamble(X_train, y_train, X_test, 10, depth, leafs)
    accuracy = evaluate_pred(y_test, y_pred)['accuracy']
    print(f"Accuracy ensamble:\t {accuracy}")
    
    # Create different plots...
    ex5(X_train, y_train, X_test, y_test, depth, leafs)
