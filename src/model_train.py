
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier 
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2)}

def model_build(X_train_sc, X_val_sc, y_train, y_val):
    # Model Building
    try:

        # Logistic Regression
        LG = LogisticRegression()
        LG.fit(X_train_sc, y_train)
        y_pred_lg = LG.predict(X_val_sc)

        # Gaussian NaiveBayes
        NB = GaussianNB()
        NB.fit(X_train_sc, y_train)
        y_pred_nb = NB.predict(X_val_sc)

        # # Support Vector Classifier
        SVM = SVC(C=0.8, kernel='rbf', probability=True)
        SVM.fit(X_train_sc, y_train)
        y_pred_svm = SVM.predict(X_val_sc)

        # KNN
        n = list(np.arange(3,20,2))
        acc = []
        for k in n:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_sc, y_train)
            # predict the response
            y_pred = knn.predict(X_val_sc)
            # evaluate accuracy
            scores = accuracy_score(y_val, y_pred)
            acc.append(scores)
        # changing to misclassification error
        MSE = [1 - x for x in acc]
        # determining the best 'k' value
        optimal_k = n[MSE.index(min(MSE))]
        # Training with optimal 'k'
        KNN = KNeighborsClassifier(n_neighbors=optimal_k)
        KNN.fit(X_train_sc, y_train)
        y_pred_knn = KNN.predict(X_val_sc)

        # Decision Tree Classifier
        DT = DecisionTreeClassifier()
        DT.fit(X_train_sc, y_train)
        y_pred_dt = DT.predict(X_val_sc)

        # Bagging Classifier
        BG = BaggingClassifier()
        BG.fit(X_train_sc, y_train)
        y_pred_bg = BG.predict(X_val_sc)

        # Random Forest Classifier
        RF = RandomForestClassifier()
        RF.fit(X_train_sc, y_train)
        y_pred_rf = RF.predict(X_val_sc)

        # Gradient Boosting Classifier
        GB = GradientBoostingClassifier()
        GB.fit(X_train_sc, y_train)
        y_pred_gb = GB.predict(X_val_sc)

        # Ada Boost Classifier
        AB = AdaBoostClassifier()
        AB.fit(X_train_sc, y_train)
        y_pred_ab = AB.predict(X_val_sc)

        return y_pred_lg, y_pred_nb, y_pred_svm, y_pred_knn, y_pred_dt, y_pred_bg, y_pred_rf, y_pred_gb, y_pred_ab, 200

    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 'error', 'error', 'error', 'error', 'error', 'error', 'error', 'error', 400 