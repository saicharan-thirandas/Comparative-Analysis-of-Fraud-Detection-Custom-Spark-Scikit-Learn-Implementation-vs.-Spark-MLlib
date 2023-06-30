# -*- coding: utf-8 -*-
#pyspark and other libraries
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
import pandas as pd
import numpy as np
import random
from collections import Counter

#Sklearn Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error


#Bucket path to read the input
s3_path = "s3://cs6240-project-bucket-hajera-pyspark/input/fraud_dataset.csv"


#Function to perform bagging and selection
def bagging_selection(train_data, test_data, target):
    bags = []
    bag_features = []

    num_samples, num_columns = train_data.shape
    columns_to_exclude = [target]

    #Randomly sample few columns to select.
    columns_to_select = [col for col in np.random.choice(train_data.columns, random.randint(3, num_columns)) if col not in columns_to_exclude]
    columns_to_select.append(columns_to_exclude[0])

    # Sample the selected columns
    train_data = train_data[columns_to_select]
    test_data = test_data[columns_to_select]

    #Randomlys sample certain no of rows between 50% to 80%
    train_data = train_data.sample(frac= random.uniform(0.5, 0.8) , random_state=42)

    return train_data,test_data , columns_to_select, train_data.count()


# Function to run model on individual partition
def run_models_on_all_partitons(model):

    # Get the broadcasted training dataset
    train_data = broadcast_train.value

    #Add test data
    test_data = broadcast_test.value

    if bagging:
      train_data,test_data,columns_to_select,count = bagging_selection(train_data, test_data , "target")

    # split data into features and target
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    #find the same X and Y for the Test data.
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    # train and evaluate each model on the partition
    results = []
    # Send input parameters to Decisiont tree classifier.
    model[1].fit(X_train, y_train )
    y_pred =  model[1].predict(X_test)

    return y_pred


# Custom partitioner class
class KeyValuePartitioner:
    def get_partition(self, key):
        return key


# Create a SparkSession
spark = SparkSession.builder \
    .appName("BroadcastAndRunModels") \
    .getOrCreate()

#df_all - current df_train to df_all
df_all = pd.read_csv(s3_path)
df_all = df_all.rename(columns={'isFraud': 'target'})


df_all["type"] = df_all["type"].astype('category')
df_all["type"] = df_all["type"].cat.codes

columns_to_drop = ["nameDest", "nameOrig", "step"]

# Split the dataset into training and testing sets
train_ratio = 0.8
df_train, df_test = train_test_split(df_all, train_size=train_ratio, random_state=42)

df_train = df_train.drop(columns_to_drop, inplace=False, axis=1)
df_test = df_test.drop(columns_to_drop, inplace=False, axis=1)

# Broadcast the training dataset

# Broadcast both df_train and df_test
broadcast_train = spark.sparkContext.broadcast(df_train)
broadcast_test = spark.sparkContext.broadcast(df_test)

# Get the broadcasted training and testing dataset
train_data = broadcast_train.value
test_data = broadcast_test.value

# split data into features and target
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
#find the same X and Y for the Test data.
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# class weights code
class_weights = len(y_train) / (2 * np.bincount(y_train))
class_weights = dict(enumerate(class_weights))



# Inidivdual Model configuration for the Voting Ensemble Classifier

lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, class_weight=class_weights)
lr2 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight=class_weights)
lr3 = LogisticRegression(penalty='l1', solver='liblinear', C=1, class_weight=class_weights)
lr4 = LogisticRegression(penalty='l1', solver='liblinear', C=10, class_weight=class_weights)
lr5 = LogisticRegression(penalty='l1', solver='liblinear', C=100, class_weight=class_weights)
lr6 = LogisticRegression(penalty='l1', solver='liblinear', C=1000, class_weight=class_weights)

lr7 = LogisticRegression(penalty='l2', C=0.01, class_weight=class_weights)
lr8 = LogisticRegression(penalty='l2', C=0.1, class_weight=class_weights)
lr9 = LogisticRegression(penalty='l2', C=1, class_weight=class_weights)
lr10 = LogisticRegression(penalty='l2', C=10, class_weight=class_weights)
lr11 = LogisticRegression(penalty='l2', C=100, class_weight=class_weights)
lr12 = LogisticRegression(penalty='l2', C=1000, class_weight=class_weights)

lr13 = LogisticRegression(penalty='elasticnet', solver='saga', C=0.01, l1_ratio=0.5, class_weight=class_weights)
lr14 = LogisticRegression(penalty='elasticnet', solver='saga', C=0.1, l1_ratio=0.5, class_weight=class_weights)
lr15 = LogisticRegression(penalty='elasticnet', solver='saga', C=1, l1_ratio=0.5, class_weight=class_weights)
lr16 = LogisticRegression(penalty='elasticnet', solver='saga', C=10, l1_ratio=0.5, class_weight=class_weights)
lr17 = LogisticRegression(penalty='elasticnet', solver='saga', C=100, l1_ratio=0.5, class_weight=class_weights)
lr18 = LogisticRegression(penalty='elasticnet', solver='saga', C=1000, l1_ratio=0.5, class_weight=class_weights)

dtc1 = DecisionTreeClassifier(criterion="gini", class_weight=class_weights, max_depth=5, min_samples_split=10)
dtc2 = DecisionTreeClassifier(criterion="gini", class_weight=class_weights, max_depth=20, min_samples_split=50)
dtc3 = DecisionTreeClassifier(criterion="gini", class_weight=class_weights, max_depth=35, min_samples_split=40)
dtc4 = DecisionTreeClassifier(criterion="gini", class_weight=class_weights, max_depth=70, min_samples_split=24)

dtc5 = DecisionTreeClassifier(criterion="entropy", class_weight=class_weights, max_depth=5, min_samples_split=10)
dtc6 = DecisionTreeClassifier(criterion="entropy", class_weight=class_weights, max_depth=20, min_samples_split=50)
dtc7 = DecisionTreeClassifier(criterion="entropy", class_weight=class_weights, max_depth=35, min_samples_split=40)
dtc8 = DecisionTreeClassifier(criterion="entropy", class_weight=class_weights, max_depth=70, min_samples_split=24)

# 'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), 'C':[1, 10]
# svc1 = SVC(kernel="linear" , C=1, class_weight=class_weights)
# svc2 = SVC(kernel="linear" , C=10, class_weight=class_weights)
# svc3 = SVC(kernel="linear" , C=100, class_weight=class_weights)

# svc4 = SVC(kernel="poly" , C=1, class_weight=class_weights)
# svc5 = SVC(kernel="poly" , C=10, class_weight=class_weights)
# svc6 = SVC(kernel="poly" , C=100, class_weight=class_weights)

# svc7 = SVC(kernel="rbf" , C=1, class_weight=class_weights)
# svc8 = SVC(kernel="rbf" , C=10, class_weight=class_weights)
# svc9 = SVC(kernel="rbf" , C=100, class_weight=class_weights)

# svc10 = SVC(kernel="sigmoid" , C=1, class_weight=class_weights)
# svc11 = SVC(kernel="sigmoid" , C=10, class_weight=class_weights)
# svc12 = SVC(kernel="sigmoid" , C=100, class_weight=class_weights)

# svc13 = SVC(kernel="precomputed" , C=1, class_weight=class_weights)
# svc14 = SVC(kernel="precomputed" , C=10, class_weight=class_weights)
# svc15 = SVC(kernel="precomputed" , C=100, class_weight=class_weights)

# ensemnble 1
voting_classifier_models_with_cwts = [lr1,lr2,lr3,lr4,lr5,lr6,lr8,lr9,lr10,lr11,lr12,lr13,lr14,lr15,lr16,lr17,lr17,
                                      dtc1,dtc2,dtc3,dtc4,dtc5,dtc6,dtc7,dtc8]
                                      # svc1,svc2,svc3,svc4,svc5,svc6,svc7,svc8,svc9,svc10,svc11,svc12]#,svc13,svc14,svc15



# Inidivdual Model configuration for the Random Forest Classifier
base_decision_tree_confiruation  = [DecisionTreeClassifier(max_depth =20, min_samples_split= 5, class_weight=class_weights)]
print(base_decision_tree_confiruation)


# Configurae the base classifiers and count here -
no_of_trees = 50
random_forest_classifier_models = base_decision_tree_confiruation*no_of_trees

# Ensemble Type to be execulted
model_list = random_forest_classifier_models
bagging = True



#change everything according to model_params
no_of_models = len(model_list)

# converts [model1,model2,model3...] to [(0,model1) , (1,modl2)...]
model_with_key_index = [ (index,value) for index, value in enumerate(model_list) ]
model_rdd = spark.sparkContext.parallelize(model_with_key_index)


#Apply the partition on the index - assigns (n,modeln) to nth partition
custom_partitioner = KeyValuePartitioner()
model_rdd_repar = model_rdd.partitionBy(numPartitions= no_of_models, partitionFunc=custom_partitioner.get_partition)

#Use model to run on each partition
y_pred = model_rdd_repar.map(run_models_on_all_partitons)

partition_count = model_rdd_repar.getNumPartitions()
print("RDD Partition Count:", partition_count)

results_list = y_pred.collect()

#voting classifier
y_test = df_test["target"]


#Transpose the results for majority prediction
transposed__result_list = list(map(list, zip(*results_list)))

#Perform voting and pick the majority predction
majority_predictions = [Counter(sublist).most_common(1)[0][0] for sublist in transposed__result_list]



#Perform error analysis.
y_pred= majority_predictions
y_actual = y_test

# Confusion Matrix
cm = confusion_matrix(y_actual, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_actual, y_pred)
print("Accuracy:", accuracy)

# F1 Score
f1 = f1_score(y_actual, y_pred)
print("F1 Score:", f1)

# False Positive Rate (FPR) and True Positive Rate (TPR)
fpr, tpr, _ = roc_curve(y_actual, y_pred)
print("False Positive Rate (FPR):", fpr[1])
print("True Positive Rate (TPR):", tpr[1])

# False Negative (FP) and True Negative (TP)
tn = cm[0, 0]
fn = cm[1, 0]
print("False Negative (FN):", fn)
print("True Negative (TN):", tn)

# False Positive (FP) and True Positive (TP)
tp = cm[1, 1]
fp = cm[0, 1]
print("False Positive (FN):", fp)
print("True Positive (TN):", tp)

# Other values (precision, recall, specificity)
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)