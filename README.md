# Machine-Learning
Machine learning and Data Mining for Breast Cancer Prediction : using WDBC (Wisconsin Breast Cancer original dataset). 
In this project i use WEKA tool and run five machine learning algorithms to compare the performance these are Naive Bayes, Decision tree j48, Multilayer perceptron,
Random forest and Support Vector Machine ( SVM ). By using some comparison metric like precision, recall, f-measure, roc area, prc are compare
these algorithms and find out that Naive Bayes giving the best result. 

The by using python implement Naive Bayes using python scikit-learn library which is very popular to data scientists and very comfortable
to use.

Performance measurment Tasks are: 
1. Split dataset to getting training and testing dataset part.
2. Build model by using train dataset.
3. predict value using test dataset.
4. compare these predicted class value and actual class value and get performance.

Dataset redundancy removing tasks are: 
Then the point come out to improve the performance. By removing redundant features of dataset if any we can do that.so i did,
1. Univariate Selection method to find out the best features.
2. remove the low score feature because it is the most redundant feature which decreasing the performance of classifier.
3. check the performance of change dataset and get improved performance of Naive Bayes.

Here i use two methods for performance check: 1.Cross validation 2. spilting
66.6% spliting, 85.5% spliting, 5-fold, 10-fold and 15-fold cross validation. 


