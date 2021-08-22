# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def _split_data_into_sets():
    df = pd.read_csv("bechdel_csv_for_predictions.csv")
    y = df['Passed Test']
    df = df.drop(columns=['Passed Test'])
    return train_test_split(df, y, test_size=0.33, random_state=42)


def _rbf_kernel_svm(X_train, y_train, models):
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    models.append(("Kernel SVM", clf))


def _logistic_regression(X_train, y_train, models):
    clf = LogisticRegression(solver="liblinear")
    clf.fit(X_train, y_train)
    models.append(("Logistic Regression", clf))


def _decission_tree_depth_3(X_train, y_train, models):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    models.append(("Decission Tree, depth 3", clf))


def _decission_tree_depth_5(X_train, y_train, models):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    models.append(("Decission Tree, depth 5", clf))


def _5_nearest_neighbors(X_train, y_train, models):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    models.append(("K Nearest Neighbors, default=5", clf))


def _2_nearest_neighbors(X_train, y_train, models):
    clf = KNeighborsClassifier(2)
    clf.fit(X_train, y_train)
    models.append(("K Nearest Neighbors, 2", clf))


def _10_nearest_neighbors(X_train, y_train, models):
    clf = KNeighborsClassifier(10)
    clf.fit(X_train, y_train)
    models.append(("K Nearest Neighbors, 10", clf))


if __name__ == '__main__':
    # creating data model and predictions stage
    X_train, X_test, y_train, y_test = _split_data_into_sets()
    fig, ax = plt.subplots()
    models = []
    _rbf_kernel_svm(X_train, y_train, models)
    _logistic_regression(X_train, y_train, models)
    _decission_tree_depth_3(X_train, y_train, models)
    _decission_tree_depth_5(X_train, y_train, models)
    _5_nearest_neighbors(X_train, y_train, models)
    model_displays = {}
    for name, clf in models:
        model_displays[name] = plot_roc_curve(clf, X_test, y_test, ax=ax, name=name)
    _ = ax.set_title("ROC curve")
    plt.show()