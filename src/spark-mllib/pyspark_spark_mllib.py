# -*- coding: utf-8 -*-

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as function
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession


spark=SparkSession.builder.getOrCreate()

s3_path = "s3://cs6240-project-bucket-saicharan-pyspark/input/fraud_dataset.csv"

df=spark.read.csv(s3_path, inferSchema=True, header=True)


#check the schema
df.printSchema()

df.head(10)

columns_to_drop = ["nameDest", "nameOrig", "step"]
df = df.drop(*columns_to_drop)

input_col = "type"
output_col = "type_id"


string_indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
df = string_indexer.fit(df).transform(df)

df.head(10)

columns_to_drop = ["type"]
df = df.drop(*columns_to_drop)

columns_to_assemble = [col for col in df.columns if col not in ["isFraud"]]

vector_assembler=VectorAssembler(
    inputCols=columns_to_assemble, outputCol='assembled_features'
)
df = vector_assembler.transform(df)

df.head(10)

#randomly split test and training data set
train, test=df.randomSplit([0.8, 0.2], seed=22)
train.count()

"""#dropping unrequired columns"""

test.groupBy('isFraud').count().show()

train_data = train.select(
    function.col('assembled_features').alias('features'),
    function.col('isFraud').alias('label')
)

test_data = test.select(
    function.col('assembled_features').alias('features'),
    function.col('isFraud').alias('label')
)

model1=DecisionTreeClassifier()


# Set the parameters
model1.setImpurity("gini")  # Impurity measure ("gini" or "entropy")
model1.setMaxDepth(10)  # Maximum depth of the tree
model1.setMaxBins(32)  # Maximum number of bins

model1.fit(train_data)

model2=RandomForestClassifier()
# Set the parameters
model2.setImpurity("gini")
model2.setMaxDepth(5)
model2.setMaxBins(32)
model2.setNumTrees(100)

model2.fit(train_data)

dtPredictions1 = model1.transform(test_data)
dtPredictions2 = model2.transform(test_data)

dtPredictions1.head(10)

y_pred = dtPredictions1.select('prediction').rdd.flatMap(lambda x: x).collect()
y_test = dtPredictions1.select('label').rdd.flatMap(lambda x: x).collect()



# Evaluate the model
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = evaluator.evaluate(dtPredictions1)
print("Accuracy:", accuracy)

y_pred = dtPredictions2.select('prediction').rdd.flatMap(lambda x: x).collect()
y_test = dtPredictions2.select('label').rdd.flatMap(lambda x: x).collect()


# Evaluate the model
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = evaluator.evaluate(dtPredictions2)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve

y_pred= y_pred
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

# Other values (precision, recall, specificity)
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)