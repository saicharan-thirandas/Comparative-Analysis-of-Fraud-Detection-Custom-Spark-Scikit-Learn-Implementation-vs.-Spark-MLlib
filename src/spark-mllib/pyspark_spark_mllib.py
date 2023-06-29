# -*- coding: utf-8 -*-
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as function
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve
from pyspark.sql import SparkSession
import time


# Your program code goes here
# Replace the following code with your actual program
for i in range(1000000):
    _ = i * i


def caluculate_metrics_on_prediction(y_pred,y_actual):
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

    # Other values (precision, recall, specificity)
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)




s3_path = "s3://cs6240-project-bucket-saicharan-pyspark/input/fraud_dataset.csv"

spark=SparkSession.builder.getOrCreate()

df=spark.read.csv(s3_path, inferSchema=True, header=True)

df = df.limit(1000)
#check the schema
df.printSchema()

columns_to_drop = ["nameDest", "nameOrig", "step"]
df = df.drop(*columns_to_drop)

input_col = "type"
output_col = "type_id"

#String Indexer transformation
string_indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
df = string_indexer.fit(df).transform(df)

columns_to_drop = ["type"]
df = df.drop(*columns_to_drop)

#Required columns to assemble for final training
columns_to_assemble = [col for col in df.columns if col not in ["isFraud"]]
vector_assembler=VectorAssembler( inputCols=columns_to_assemble, outputCol='assembled_features' )
df = vector_assembler.transform(df)


#randomly split test and training data set
train, test=df.randomSplit([0.8, 0.2], seed=22)
train.count()

"""#dropping unrequired columns"""
test.groupBy('isFraud').count().show()


# Selecting training and testing data
train_data = train.select(
    function.col('assembled_features').alias('features'),
    function.col('isFraud').alias('label')
)

test_data = test.select(
    function.col('assembled_features').alias('features'),
    function.col('isFraud').alias('label')
)



print(" Decision Tree Classifier Results : ")
# Record the start time
start_time = time.time()

#Run Decision Tree - with parameters
model_decision_tree=DecisionTreeClassifier()
# Set the parameters
model_decision_tree.setImpurity("gini")  # Impurity measure ("gini" or "entropy")
model_decision_tree.setMaxDepth(10)  # Maximum depth of the tree
model_decision_tree.setMaxBins(32)  # Maximum number of bins

model_decision_tree = model_decision_tree.fit(train_data)

df_predictions_decison_tree = model_decision_tree.transform(test_data)
y_pred = df_predictions_decison_tree.select('prediction').rdd.flatMap(lambda x: x).collect()
y_test = df_predictions_decison_tree.select('label').rdd.flatMap(lambda x: x).collect()

caluculate_metrics_on_prediction(y_pred,y_test)

# Record the end time
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time
# Print the start and end time
print("Start Time: ", time.ctime(start_time))
print("End Time: ", time.ctime(end_time))
print("Execution Time: ", execution_time, "seconds")

#Run Random Forest Classifier - with parameters
print(" ######################################## ")

print(" Random Forest Classifier  Results : ")

start_time = time.time()

model_random_forest=RandomForestClassifier()
# Set the parameters
model_random_forest.setImpurity("gini")
model_random_forest.setMaxDepth(30)
model_random_forest.setMaxBins(32)
model_random_forest.setNumTrees(20)

model_random_forest = model_random_forest.fit(train_data)

df_predictions_random_forest =  model_random_forest.transform(test_data)

y_pred = df_predictions_random_forest.select('prediction').rdd.flatMap(lambda x: x).collect()
y_test = df_predictions_random_forest.select('label').rdd.flatMap(lambda x: x).collect()

caluculate_metrics_on_prediction(y_pred,y_test)

# Record the end time
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time
# Print the start and end time
print("Start Time: ", time.ctime(start_time))
print("End Time: ", time.ctime(end_time))
print("Execution Time: ", execution_time, "seconds")




