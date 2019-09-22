    # -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:09:11 2019

@author: Animesh
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn import metrics


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    
def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def loadCsv_for_weka(filename,pred):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)

	cnt = 0
	for i in range(len(pred)):
		if(int(pred[i]) == int(dataset[i][10])):
			cnt = cnt + 1;

	print("Score of matching weka result and this algorithm predicted value :: ",cnt/len(pred)*100,'%');


def trainData(dataset):
	train = []
	for i in range(len(dataset)):
		new = []
		for j in range(9):
			new.append(dataset[i][j])
		train.append(new)

	return train



def classMaking(dataset):
	test_class = {}
	for i in range(len(dataset)):
		test_class[i] = dataset[i][9]
	return test_class	

def weka_getting_class_value(filename):
	dataset = loadCsv_for_weka(filename)
	weka = {}
	counter = 0
	for i in range(len(dataset)):
		weka[i] = float(dataset[i][10])
		counter = counter + 1
		print(dataset[i][10] in dataset)
		print(counter)
	return weka 


def push(pred):
	print(pred[1])
	with open('D:/4-1/Thesis/NAIVE BAYES NEW/TestData - Copy.csv','r') as csvinput:
		with open('D:/4-1/Thesis/NAIVE BAYES NEW/test_file_with_predicted_classValue.csv', 'w') as csvoutput:
		    writer = csv.writer(csvoutput, lineterminator='\n')
		    reader = csv.reader(csvinput)

		    all = []
		    i = 0

		    for row in reader:
		        row.append(pred[i])
		        i = i+1
		        all.append(row)

		    writer.writerows(all)

def wka_match(pred,weka):
	cnt = 0;
	for i in range(len(pred)):
		if(pred[i] == weka[i][10]):
			cnt = cnt+1;

	return cnt/len(pred);

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def precision(tn, fp, fn, tp):
    return tp / (tp + fp)

def recall (tn, fp, fn, tp):
    return tp / (tp + fn)

def confusionMatrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def rd(data):
    data = round(data,3)
    return data

def crossValidation(data, target):
    import statistics
    data = np.array(data)
    target = np.array(target)
    folds = [5,10,15]
    for fold in folds:
        cv = KFold(n_splits = fold, random_state = 0, shuffle = False)
        scores = []
        tn = 0 
        fp = 0
        fn = 0 
        tp = 0
        for train_index, test_index in cv.split(data,target):
            xtrain, xtest, ytrain, ytest = data[train_index], data[test_index], target[train_index], target[test_index]
            model = GaussianNB()
            model.fit(xtrain, ytrain)
            p = model.predict(xtest)
            scores.append(accuracy_score(p, ytest))
            tn1, fp1, fn1, tp1 = confusionMatrix(ytest, p)
            
            tn += tn1
            fp += fp1
            fn += fn1
            tp += tp1
            
        pre = precision(tn, fp, fn, tp) 
        re = recall( tn, fp, fn, tp) 
        pre = rd(pre)
        re = rd(re)
        print('for ', fold , ' cross validation result: ', rd(statistics.mean(scores)), ' precision : ',pre ,' recall : ', re)


    return pre, re, scores

def split(data, target, testSize):
    xtrain,xtest,ytrain,ytest = train_test_split(data,target,test_size = testSize,random_state = 42) #85.5% split
    model = GaussianNB()
    model.fit(xtrain, ytrain)
    p = model.predict(xtest)
    score = accuracy_score(p,ytest)*100
    tn, fp, fn, tp = confusionMatrix(ytest, p)
    pre = precision(tn, fp, fn, tp) 
    re = recall( tn, fp, fn, tp)
    #round 
    pre = rd(pre)
    score = rd(score)
    re = rd(re)
    return p, ytest, score, pre, re

def confusin_matrix_pic(p, ytest):
    cm = metrics.confusion_matrix(ytest, p)
    import seaborn as sn
    sn.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],)
    plt.ylabel('True label',fontsize=12)
    plt.xlabel('Predicted label',fontsize=12)
    
def roc(y_true, y_score):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(y_true, y_score)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

def main():    
    filename = 'D:/4-1/Thesis/NAIVE BAYES NEW/TrainData.csv'
    test_file = 'D:/4-1/Thesis/NAIVE BAYES NEW/TestData.csv'
    weka_result_file = 'D:/4-1/Thesis/NAIVE BAYES NEW/weka_result.csv'
    dataset = loadCsv(filename)#the train dataset with class 
    test_dataset = loadCsv(test_file)#test dataset 
    train_dataset = trainData(dataset) #the train dataset without class 
    class_value = classMaking(dataset)
    x_train , x_test , y_train,y_test = train_test_split(train_dataset,class_value,test_size = 0.1432,random_state = 42)

    #make model of WDBC 
    model = GaussianNB()
    model.fit(x_train,y_train)
    pred = model.predict(test_dataset)

    print("Accuracy = ",(accuracy_score(y_test,pred)*100),"%")
    al_pred = model.predict(x_test)
    print("Accuracy of naive-bayes algorithm : ",round(accuracy_score(al_pred,y_test),2)*100,"%")

    data = pd.read_csv(filename)
    print(data.isnull().sum())
    X = data.iloc[:,0:9]
    Y = data.iloc[:,-1]
    p, ytest, acc_score, pre, re = split(X, Y, 0.334)  #66.6%split
    print("66.6% split Accuracy: ",acc_score,"%", ' precision: ', pre, ' recall: ', re)
#     confusion_matrix_pic(p, ytest)
#     confusin_matrix_pic(p, ytest)

    p, ytest, acc_score, pre, re = split(X, Y, 0.1450)  #85.5%split
    print("85.5% split Accuracy: ",acc_score,"%", ' precision: ', pre, ' recall: ', re)
#     confusin_matrix_pic(p, ytest)

        #@.. cross validation
    print('Cross Validation of 10 Attributes')
    pre, re, scores = crossValidation(X, Y)

    #feature extraction and Selection Method : Univariate Selection
    bestfeatures = SelectKBest(score_func=chi2, k=9)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Attributes','Score']  #naming the dataframe columns
    print(featureScores.nlargest(9,'Score'))  #print features with scores 
    
#     .......................#remove 'single epithelial cell'................
    data = data.drop('\t1\t.5', axis = 1)
    
    #accuracy for processed dataset
    target = data.iloc[:,-1]
    data = data.iloc[:,0:8]
    p, ytest,acc_score, pre, re = split(data, target, 0.334)  #66.6%split
    print("66.6% split Accuracy: ",acc_score,"%", ' precision: ', pre, ' recall: ', re)
#     roc(ytest, p)

    class_names = [1,2,3,4,5,6,7,8,9]
    p, ytest, acc_score, pre, re = split(data, target, 0.1450)  #85.5%split
    print("85.5% split Accuracy: ",acc_score,"%", ' precision: ', pre, ' recall: ', re)
    #@ cross validation of modified dataset : precision, recall 
    pre, re, scores = crossValidation(data, target)
    

main()
