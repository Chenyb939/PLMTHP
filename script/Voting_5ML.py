import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import math
import torch
import pathlib
import argparse
import AAC_DPC_Feature_Extraction
import MLs_Parameter_Tuning
import ADE_Combination
import ESM_Extraction
import warnings
warnings.filterwarnings("ignore")

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


#PLMTHP independent test prediction
def voting(Xt,knnparam,mlpparam,lnc,lngamma,rbfc,rbfgamma,weight1,weight2,weight3,weight4,weight5,out_path):
    KNN = KNeighborsClassifier(n_neighbors=knnparam)
    MLP = MLPClassifier(hidden_layer_sizes=(mlpparam,), random_state=0)
    NB = GaussianNB()
    SVMLN = SVC(C=lnc, kernel='linear', gamma=lngamma, probability=True, random_state=0)
    SVMRBF = SVC(C=rbfc, kernel='rbf', gamma=rbfgamma, probability=True, random_state=0)
    # Weighted voting ensemble classifiers
    estimators = [("KNN", KNN), ("MLP", MLP), ("NB", NB), ("SVMLN", SVMLN), ("SVMRBF", SVMRBF)]
    clf_weighted3 = VotingClassifier(estimators, voting="soft", weights=[weight1,weight2,weight3,weight4,weight5])
    # test
    p=clf_weighted3.predict(Xt)
    with open(os.path.join(out_path, "PLMTHP_prediction.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([p])

def create_parser():
    parser = argparse.ArgumentParser(
        description="THP prediction is made on the PLMTHP model"
    )
    parser.add_argument(
        "--trainpos",
        type=pathlib.Path,
        required=True,
        default='',
        help='Path of pos train data file'
    )
    parser.add_argument(
        "--trainneg",
        type=pathlib.Path,
        required=True,
        default='',
        help='Path of neg train data file'
    )
    parser.add_argument(
        "--test",
        type=pathlib.Path,
        required=True,
        default='',
        help='Path of test data file'
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        default='',
        help='Path of output file'
    )
    parser.add_argument(
        "-model_location",
        type=str,
        default='esm2_t36_3B_UR50D',
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "-toks_per_batch",
        type=int,
        default=8192,
        help="maximum batch size"
    )
    parser.add_argument(
        "-repr_layers",
        type=int,
        default=36,
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "-include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        default="mean",
        help="specify which representations to return",
    )
    parser.add_argument(
        "-nogpu",
        action="store_true",
        default="true",
        help="Do not use GPU even if available"
    )
    return parser



parser = create_parser()
args = parser.parse_args()
ESM_Extraction.esmpos(args)
ESM_Extraction.esmneg(args)

'''
###Train
#extract AAC and DPC features
AAC_DPC_Feature_Extraction.extract_train(r"args.trainpos",r"args.trainneg")

#extract ESM features


#Combine AAC、ADE and ESM
AAC+DPC+ESM_Combination.combine_train()

#Classifiers' parameters tuning
MLs_Parameter_Tuning.param_tuning()

#Train data
pos_ade = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pos_ade.pt')
neg_ade = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'neg_ade.pt')
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




###Test
#extract AAC and DPC features
AAC_DPC_Feature_Extraction.extract_test(r"args.test")

#extract ESM features


#Test data
ade = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ade.pt')
all_data = ade.numpy()
Xt = all_data
del ade,all_data


params = np.loadtxt(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train_Best_Params.csv"),"rb"),delimiter=",",skiprows=0)
print(params)
print(params[0])
voting(X,y,Xt,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10],out_path)


'''