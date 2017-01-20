"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p


#loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')

def main():

    print ("loading data..")
    traindata = list(np.array(p.read_table('train.tsv'))[:,2])
    testdata = list(np.array(p.read_table('test.tsv'))[:,2])
    y = np.array(p.read_table('train.tsv'))[:,-1]

    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                          analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

    x_all = traindata + testdata
    len_train = len(traindata)

    print "fitting pipeline"
    tfv.fit(x_all)

    ##################
    x2 = tfv.fit_transform((np.array(x_all))) #returns a matrix of [n_samples, n_features]
    print (np.array(x_all).shape)
    print (tfv.idf_.shape)
    print (x2.shape)
    ##################

    print "transforming data"
    x_all = tfv.transform(x_all)

    x_train = x_all[:len_train]
    x_test = x_all[len_train:]

    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, x_train, y, cv=20, scoring='roc_auc'))

    print "training on full data"
    rd.fit(x_train,y)
    prediction = rd.predict_proba(x_test)[:,1]
    testfile = p.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    prediction_df = p.DataFrame(prediction, index=testfile.index, columns=['label'])
    prediction_df.to_csv('benchmark.csv')
    print "submission file created.."

if __name__ == "__main__":
  main()
