#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:59:34 2023

@author: lucaromeo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, auc, multilabel_confusion_matrix


def perform_Kfold_cv(X, Y, seed_value, model_name, hyperparameters):
    
    rnd_state = seed_value


    # Define the model with the given hyperparameters
    if model_name == 'XGBoost':
        base_classifier = XGBClassifier(random_state=rnd_state)
        param_grid=hyperparameters
    elif model_name == 'Random Forest':
        base_classifier = RandomForestClassifier(random_state=rnd_state)
        param_grid=hyperparameters
    elif model_name == 'SVM L1L2':
        base_classifier = SGDClassifier(random_state=rnd_state, penalty='elasticnet')
        param_grid=hyperparameters
    elif model_name == 'SVM L2':
        base_classifier = SGDClassifier(random_state=rnd_state, penalty='l2')
        param_grid=hyperparameters
    elif model_name == 'KNN':
        base_classifier = KNeighborsClassifier()
        param_grid=hyperparameters
    elif model_name == 'MLP':
        base_classifier = MLPClassifier(random_state=rnd_state)
        param_grid=hyperparameters
    elif model_name == 'Naive Bayes':
        base_classifier = GaussianNB()
        param_grid=hyperparameters
    else:
        raise ValueError('Invalid model name.')
        
        
    #if model_name not in ['Naive Bayes']:
    pipeline = Pipeline([('scaler', StandardScaler()),('multi_output', MultiOutputClassifier(base_classifier, n_jobs=-1))])
    #else:
    #    pipeline = Pipeline([('multi_output', MultiOutputClassifier(base_classifier, n_jobs=-1))])

    #scoring = make_scorer(balanced_accuracy_score)
    scoring = make_scorer(f1_score, average='macro')

    #outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd_state)
    #inner_cv = KFold(n_splits=5, shuffle=True, random_state=rnd_state)


    # hold-out procedure
    # X_train=X.iloc[:1200]
    #X_test=X.iloc[1200:]
    #y_train=Y.iloc[:1200]
    #y_test=Y.iloc[1200:]

    # added - use train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

       
       # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
       # y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]


#    clf = GridSearchCV(pipeline, param_grid, cv=inner_cv, refit=True)
    clf = GridSearchCV(pipeline, param_grid, cv=5, refit=True, scoring=scoring)

    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    print(best_params)
    best_estimator = clf.best_estimator_
     
    # Predict labels for the test data
    y_pred = best_estimator.predict(X_test)
     
     
    label_cm = multilabel_confusion_matrix(y_test, y_pred)
     
     
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(label_cm[0,:,:], annot=True, fmt=".1f", ax=ax)
    ax.set_title('Confusion Matrix Target Cassette')
    ax.xaxis.set_ticklabels(['Cassette NO','Cassette YES'])
    ax.yaxis.set_ticklabels(['Cassette NO','Cassette YES'])
    #plt.savefig('figure2806/cmMatrixCassetteW6.pdf', bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = label_cm[0,:,:].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba0=(sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba0 )
     
     
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(label_cm[1,:,:], annot=True, fmt=".1f", ax=ax)
    ax.set_title('Confusion Matrix Target CT')
    ax.xaxis.set_ticklabels(['CT NO','CT YES'])
    ax.yaxis.set_ticklabels(['CT NO','CT YES'])
    #plt.savefig('figure2806/cmMatrixCTW6.pdf', bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = label_cm[1,:,:].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba1=(sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba1 )
     
     
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(label_cm[2,:,:], annot=True, fmt=".1f", ax=ax)
    ax.set_title('Confusion Matrix Target NE')
    ax.xaxis.set_ticklabels(['NE NO','NE YES'])
    ax.yaxis.set_ticklabels(['NE NO','NE YES'])
    #plt.savefig('figure2806/cmMatrixNEW6.pdf', bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = label_cm[2,:,:].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba2=(sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba2 )
     
     
     
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(label_cm[3,:,:], annot=True, fmt=".1f", ax=ax)
    ax.set_title('Confusion Matrix Target NF')
    ax.xaxis.set_ticklabels(['NF NO','NF YES'])
    ax.yaxis.set_ticklabels(['NF NO','NF YES'])
    #plt.savefig('figure2806/cmMatrixNFW6.pdf', bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = label_cm[3,:,:].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba3=(sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba3 )
     
     
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(label_cm[4,:,:], annot=True, fmt=".1f", ax=ax)
    ax.set_title('Confusion Matrix Target NV')
    ax.xaxis.set_ticklabels(['NV NO','NV YES'])
    ax.yaxis.set_ticklabels(['NV NO','NV YES'])
    #plt.savefig('figure2806/cmMatrixNVW6.pdf', bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = label_cm[4,:,:].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba4=(sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba4 )
     
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(label_cm[5,:,:], annot=True, fmt=".1f", ax=ax)
    ax.set_title('Confusion Matrix Target SHUTTER')
    ax.xaxis.set_ticklabels(['SHUTTER NO','SHUTTER YES'])
    ax.yaxis.set_ticklabels(['SHUTTER NO','SHUTTER YES'])
    #plt.savefig('figure2806/cmMatrixSHUTTERW6.pdf', bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = label_cm[5,:,:].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba5=(sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba5 )