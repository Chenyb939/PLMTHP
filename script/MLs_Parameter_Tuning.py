import math
import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import csv
#Cross-Validation on the train dataset
def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)

    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:, 1]
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(0, len(test_y)):
            if test_y[i] == 1 and p[i] == 1:
                TP += 1
            elif test_y[i] == 1 and p[i] == 0:
                FN += 1
            elif test_y[i] == 0 and p[i] == 1:
                FP += 1
            elif test_y[i] == 0 and p[i] == 0:
                TN += 1
        ACC = (TP + TN) / (TP + FP + TN + FN)
        SENS = TP / (TP + FN)
        SPEC = TN / (TN + FP)
        det = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if (det == 0):
            MCC = 0
        else:
            MCC = ((TP * TN) - (FP * FN)) / det
        AUC = roc_auc_score(test_y, pr)

        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)

    return np.mean(allACC), np.mean(allSENS), np.mean(allSPEC), np.mean(allMCC), np.mean(allAUC)



# Tuning the best parameters of classifiers and evluation classifiers
def knn_patu(X,y,file):
    param = [i for i in np.arange(1, 100, dtype=int)]
    acc = np.zeros(len(param))
    sens = np.zeros(len(param))
    spec = np.zeros(len(param))
    mcc = np.zeros(len(param))
    auc = np.zeros(len(param))
    for i in range(1, len(param)):
        clf = KNeighborsClassifier(n_neighbors=int(i))
        acc[i], sens[i], spec[i], mcc[i], auc[i] = cv(clf, X, y, 10)
    choose = np.argmax(auc)
    #Record the performence of KNN
    file.write("KNN"+","+str(acc[choose]) + "," + str(sens[choose]) + "," + str(spec[choose]) + "," + str(mcc[choose]) + "," + str(roc[choose]) + "\n")
    #Record the best parameter
    return param[choose]


# Tuning the best parameters of mlp classifier and train the model
def mlp_patu(X,y,file):
    param = [2 ** i for i in np.arange(1, 9, dtype=int)]
    acc = np.zeros(len(param))
    sens = np.zeros(len(param))
    spec = np.zeros(len(param))
    mcc = np.zeros(len(param))
    auc = np.zeros(len(param))
    for i in range(0, len(param)):
        clf = MLPClassifier(hidden_layer_sizes=(param[i],), random_state=0)
        acc[i], sens[i], spec[i], mcc[i], auc[i] = cv(clf, X, y, 10)
    choose = np.argmax(auc)
    file.write("MLP"+","+str(acc[choose]) + "," + str(sens[choose]) + "," + str(spec[choose]) + "," + str(mcc[choose]) + "," + str(roc[choose]) + "\n")
    return param[choose]


def nb(X,y,file):
    NB = GaussianNB()
    acc, sens, spec, mcc, auc = cv(NB, X, y, 10)
    file.write("NB"+","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n")

# Tuning the best parameters of SVMLN classifier
def svmln_patu(X,y,file):
    clf = SVC(kernel='linear', probability=True, random_state=0)
    grid = GridSearchCV(clf, param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, scoring='roc_auc', cv=10)
    grid.fit(X, y)
    clf = SVC(C=grid.best_params_['C'], kernel='linear', gamma=grid.best_params_['gamma'], probability=True, random_state=0).fit(X, y)
    clf = clf.fit(X, y)
    acc, sens, spec, mcc, auc = cv(clf, X, y, 10)
    file.write("SVMLN"+","+str(acc) + "," + str(sens) + "," + str(spec) + "," + str(mcc) + "," + str(roc) + "\n")
    return grid.best_params_['C'],grid.best_params_['gamma']


# Tuning the best parameters of SVMRBF classifier
def svmrbf_patu(X,y,file):
    from sklearn.model_selection import GridSearchCV
    clf = SVC(kernel='rbf', probability=True, random_state=0)
    grid = GridSearchCV(clf, param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, scoring='roc_auc',cv=10)
    grid.fit(X, y)
    clf = SVC(C=10, kernel='rbf', gamma=0.1, probability=True, random_state=0).fit(X, y)
    clf = clf.fit(X, y)
    acc, sens, spec, mcc, auc = cv(clf, X, y, 10)
    file.write("SVMRBF"+","+str(acc) + "," + str(sens) + "," + str(spec) + "," + str(mcc) + "," + str(roc) + "\n")
    return grid.best_params_['C'],grid.best_params_['gamma']


#Search the best voting weight of PLMTHP
def voting_patu(X,y,k,m,lnc,lng,rbfc,rbfg,file):
    KNN = KNeighborsClassifier(n_neighbors=k)
    MLP = MLPClassifier(hidden_layer_sizes=(m,), random_state=0)
    NB = GaussianNB()
    SVMLN = SVC(C=lnc, kernel='linear', gamma=lng, probability=True, random_state=0)
    SVMRBF = SVC(C=rbfc, kernel='rbf', gamma=rbfg, probability=True, random_state=0)
    estimators = [("KNN", KNN), ("MLP", MLP), ("NB", NB), ("SVMLN", SVMLN), ("SVMRBF", SVMRBF)]
    param1 = [0.4 * i for i in np.arange(2, 5, dtype=float)]
    param2 = [0.4 * j for j in np.arange(2, 5, dtype=float)]
    param3 = [0.4 * k for k in np.arange(2, 5, dtype=float)]
    param4 = [0.4 * h for h in np.arange(2, 5, dtype=float)]
    param5 = [0.4 * p for p in np.arange(2, 5, dtype=float)]
    max=0
    param=np.zeros(5)
    for i in range(2, 5):
        for j in range(2,5):
            for k in range(2, 5):
                for h in range(2, 5):
                    for p in range(2, 5):
                        clf_weighted3 = VotingClassifier(estimators, voting="soft", weights=[param1[i], param2[j],param3[k],param4[h],param5[p]])
                        acc, sens, spec, mcc, auc = cv(clf_weighted3, X, y, 10)
                        if auc>max:
                            max=auc
                            param[0]=param1[i]
                            param[1]=param2[j]
                            param[2] = param3[k]
                            param[3] = param4[h]
                            param[4] = param5[p]
    clf_weighted3 = VotingClassifier(estimators, voting="soft",weights=[param[0], param[1], param[2], param[3], param[4]])
    acc, sens, spec, mcc, auc = cv(clf_weighted3, X, y, 10)
    file.write("PLMTHP" + "," + str(acc) + "," + str(sens) + "," + str(spec) + "," + str(mcc) + "," + str(roc) + "\n")
    return param

def param_tuning():
    # Load dataset and split the dataset(Please change your paths)
    pos_ade = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pos_ade.pt'))
    neg_ade = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neg_ade.pt'))
    pos = pos_ade.numpy()
    neg = neg_ade.numpy()
    all_data = np.concatenate((pos, neg), axis=0)
    n = all_data.shape[0]
    m = pos.shape[0]

    X = all_data
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if i < m:
            y[i] = 1
        else:
            y[i] = 0

    del pos_ade, neg_ade, pos, neg, all_data

    #Record the evluation of tain dataset
    file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train_Evaluation.csv"), "w")
    file.write("Classifiers" + "," +"ACC" + "," + "SENS"+ "," + "SPEC" + "," + "MCC"+ "," + "ROC"+ "\n")

    #Record the best parameters of classifiers
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train_Best_Params.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["KNN_n_neighbors", "MLP_hidden_layer_sizes", "SVMLN_C", "SVMLN_gamma", "SVMRBF_C","SVMRBF_gamma","Weight1","Weight2","Weight3","Weight4","Weight5"])
        k=knn_patu(X, y,file)
        m=mlp_patu(X, y,file)
        nb(X,y,file)
        lnc,lng=svmln_patu(X,y,file)
        rbfc,rbfg=svmrbf_patu(X, y,file)
        param=voting_patu(X,y,k,m,lnc,lng,rbfc,rbfg,file)
        writer.writerow([str(k),str(m),str(lnc),str(lng),str(rbfc),str(rbfg),param[0],param[1],param[2],param[3],param[4]])

    file.close()


