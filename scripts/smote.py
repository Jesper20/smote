import time
import datetime
import numpy as np
import pandas as pd 
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import smote_variants as sv


def write2File(name, scores) : 
    print(name)
    print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std()))
    print("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std()))
    
    print("F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))
    print("AUC: %0.2f (+/- %0.2f)" % (scores['test_auc'].mean(), scores['test_auc'].std()))
    print("Fit_time: %0.2f (+/- %0.2f)" % (scores['fit_time'].mean(), scores['fit_time'].std()))
    
    result = name + "," + feature_type +","
    result += sampler_name + ","
    result += str(n_instances) +  ","
    result += str(n_features) + ","
    result += "%0.5f" % (scores['test_recall'].mean()) + ","
    result += "%0.5f" % (scores['test_recall'].std()) + ","
    result += "%0.5f" % (scores['test_precision'].mean()) + ","
    result += "%0.5f" % (scores['test_precision'].std()) + ","
    result += "%0.5f" % (scores['test_f1'].mean()) + ","
    result += "%0.5f" % (scores['test_f1'].std()) + ","
    result += "%0.5f" % (scores['test_auc'].mean()) + ","
    result += "%0.5f" % (scores['test_auc'].std()) + ","
    result += "%0.5f" % (scores['fit_time'].mean()) + ","
    result += "%0.5f" % (scores['fit_time'].std()) + ","

    result += str(cv) +  ","
    result += f'{(time.time() - start_time)/3600:.2f},'
    result += note + ","
    result += str(datetime.datetime.now()) + '\n'

    f = open(outFile, "a+")
    f.write(result)
    f.close()

###########  Modify the following three variables ##############################
# select one of the features csv file
inputFile = '../datasets/dcuf.csv.gz'

# Classifiers 
names = ["Naive Bayes", "Random Forest"]
classifiers = [GaussianNB(), RandomForestClassifier()]

oversamplers = [ 
    sv.ADASYN(), sv.ADG(), sv.ADOMS(), sv.AHC(), sv.AND_SMOTE(),
    sv.ANS(), sv.ASMOBD(), sv.Assembled_SMOTE(), 
    sv.Borderline_SMOTE1(), sv.Borderline_SMOTE2(), 
    sv.CBSO(), sv.CCR(), sv.CE_SMOTE(), sv.cluster_SMOTE(),
    sv.CURE_SMOTE(), sv.DBSMOTE(), sv.DE_oversampling(), sv.distance_SMOTE(),
    sv.Edge_Det_SMOTE(), sv.G_SMOTE(), sv.Gaussian_SMOTE(), sv.Gazzah(), 
    sv.IPADE_ID(), sv.ISMOTE(), sv.Lee(), sv.LLE_SMOTE(), sv.LN_SMOTE(), 
    sv.LVQ_SMOTE(), sv.MCT(), sv.MDO(), sv.MOT2LD(), sv.MSMOTE(), sv.MSYN(), 
    sv.MWMOTE(), sv.NDO_sampling(), sv.NEATER(), sv.NRAS(), sv.NRSBoundary_SMOTE(), 
    sv.NT_SMOTE(), sv.OUPS(), sv.polynom_fit_SMOTE(), sv.ProWSyn(), sv.Random_SMOTE(), 
    sv.ROSE(), sv.Safe_Level_SMOTE(), sv.SDSMOTE(), sv.Selected_SMOTE(), sv.SL_graph_SMOTE(), 
    sv.SMMO(), sv.SMOBD(), sv.SMOTE(), sv.SMOTE_Cosine(), sv.SMOTE_D(), sv.SMOTE_ENN(), 
    sv.SMOTE_FRST_2T(), sv.SMOTE_IPF(), sv.SMOTE_OUT(), sv.SMOTE_RSB(), 
    sv.SMOTE_TomekLinks(), sv.SN_SMOTE(), sv.SOI_CJ(), sv.SOMO(), 
    sv.SPY(), sv.Stefanowski(), sv.SUNDO(), sv.Supervised_SMOTE(), sv.SVM_balance(), 
    sv.TRIM_SMOTE(), sv.V_SYNTH(), sv.VIS_RST() 
] 
sampler_names = [ 
    "ADASYN", "ADG", "ADOMS", "AHC", "AND_SMOTE",
    "ANS", "ASMOBD", "Assembled_SMOTE", 
    "Borderline_SMOTE1", "Borderline_SMOTE2", 
    "CBSO", "CCR", "CE_SMOTE", "cluster_SMOTE",
    "CURE_SMOTE", "DBSMOTE", "DE_oversampling", "distance_SMOTE",
    "Edge_Det_SMOTE", "G_SMOTE", "Gaussian_SMOTE", "Gazzah", 
    "IPADE_ID", "ISMOTE", "Lee", "LLE_SMOTE", "LN_SMOTE", 
    "LVQ_SMOTE", "MCT", "MDO", "MOT2LD", "MSMOTE", "MSYN", 
    "MWMOTE", "NDO_sampling", "NEATER", "NRAS", "NRSBoundary_SMOTE", 
    "NT_SMOTE", "OUPS", "polynom_fit_SMOTE", "ProWSyn", "Random_SMOTE", 
    "ROSE", "Safe_Level_SMOTE", "SDSMOTE", "Selected_SMOTE", "SL_graph_SMOTE", 
    "SMMO", "SMOBD", "SMOTE", "SMOTE_Cosine", "SMOTE_D", "SMOTE_ENN", 
    "SMOTE_FRST_2T", "SMOTE_IPF", "SMOTE_OUT", "SMOTE_RSB", 
    "SMOTE_TomekLinks", "SN_SMOTE", "SOI_CJ", "SOMO", 
    "SPY", "Stefanowski", "SUNDO", "Supervised_SMOTE", "SVM_balance", 
    "TRIM_SMOTE", "V_SYNTH", "VIS_RST"
]   

###########  Modify the above three variables ##############################

# evaluation result 
outFile = 'smote.csv'
# determine type of features that are being used
idx = inputFile.rfind('/') + 1
feature_type = inputFile[idx:]
print("Running " + feature_type + " features ")
# 5-fold cross validation and measures
cv = 5
#scoring = ['precision', 'recall', 'f1']
# read csv features file
df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)

# X = features, y = class label (0 or 1)
X = df.dropna(axis=1, how='all') # drop columns (axis=1) with 'all' NaN values
# get data without label
X = X.drop('classLabel', axis=1)
# y = labels
y = df['classLabel']
n_features = len(X.columns) # number of features
n_instances = len(X) # number of instances
print("No of features: " + str(n_features))
print("No of ins: " + str(n_instances))

# convert to numpy arrays
X = X.values
y = y.values

# evaluate each classifier with 85 oversamplers
for oversampler, sampler_name in zip(oversamplers, sampler_names):
    start_time = time.time()
    
    note = feature_type + " features"
    print("Running " + sampler_name)
    
    # Stratified k fold cross validation
    scores1 = { }
    fmeasures1 = []
    recalls1 = []
    precisions1 = []
    aucs1 = []
    fit_times1 = []

    scores2 = { }
    fmeasures2 = []
    recalls2 = []
    precisions2 = []
    aucs2 = []
    fit_times2 = []
    kf = StratifiedKFold(n_splits=cv)
    for train_idx, test_idx, in kf.split(X, y):
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train_oversampled, y_train_oversampled = oversampler.sample(X_train, y_train)
        
        fit_start = time.time() 
        model1 = make_pipeline(StandardScaler(), classifiers[0])
        model1.fit( X_train_oversampled, y_train_oversampled )  
        fit_times1.append(time.time() - fit_start)
        
        y_pred = model1.predict(X_test)
        fmeasures1.append(f1_score(y_test, y_pred))
        recalls1.append(recall_score(y_test, y_pred))
        precisions1.append(precision_score(y_test,y_pred))
        aucs1.append( roc_auc_score(y_test, model1.predict_proba(X_test)[:, 1]) )

        fit_start = time.time() 
        model2 = make_pipeline(StandardScaler(), classifiers[1])
        model2.fit( X_train_oversampled, y_train_oversampled )  
        fit_times2.append(time.time() - fit_start)
        
        y_pred = model2.predict(X_test)
        fmeasures2.append(f1_score(y_test, y_pred))
        recalls2.append(recall_score(y_test, y_pred))
        precisions2.append(precision_score(y_test,y_pred))
        aucs2.append( roc_auc_score(y_test, model2.predict_proba(X_test)[:, 1]) )
        
    scores1['test_recall'] = np.array(recalls1)
    scores1['test_precision'] = np.array(precisions1)
    scores1['test_f1'] = np.array(fmeasures1)
    scores1['test_auc'] = np.array(aucs1)
    scores1['fit_time'] = np.array(fit_times1)

    write2File(names[0], scores1)

    scores2['test_recall'] = np.array(recalls2)
    scores2['test_precision'] = np.array(precisions2)
    scores2['test_f1'] = np.array(fmeasures2)
    scores2['test_auc'] = np.array(aucs2)
    scores2['fit_time'] = np.array(fit_times2)

    write2File(names[1], scores2)
    