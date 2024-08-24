# Comparative Analysis of Fraud Detection: Custom Spark + Scikit-Learn Implementation vs. Spark MLlib

## Project Overview

This project focuses on developing and evaluating scalable machine learning models to detect fraudulent transactions in digital payment systems. With an increasing need for banks to proactively identify and block fraudulent activities, our project aims to enhance their ability to protect clients and combat global money laundering.

### Key Approaches
1. **Custom Implementation**: Leveraging Spark's in-memory processing and scikit-learn for classification and prediction ensembles.
2. **Spark MLlib**: Utilizing Spark MLlib's Decision Tree and Random Forest libraries for classification and prediction tasks.

### Dataset
- **Source**: Simulated mobile money transactions dataset from Kaggle.
- **Size**: 6,362,620 records with no missing values.
- **Features**: `step`, `type`, `amount`, `nameOrig`, `oldBalanceOrg`, `newBalanceOrig`, `nameDest`, `oldBalanceDest`, `newBalanceDest`, `isFraud`, and `isFlaggedFraud`.

## Comparative Analysis

### Speed
- **Custom Implementation**: Demonstrated faster execution times, particularly for the Decision Tree classifier, due to optimized scikit-learn implementations.
- **Spark MLlib**: While designed for big data and distributed computing, Spark MLlib incurs additional time for feature transformations, leading to longer execution times for smaller datasets.

### Accuracy
- **Custom Implementation**: Achieved high accuracy, particularly with ensemble models using scikit-learn.
- **Spark MLlib**: Comparable accuracy, with potential advantages when applying class imbalance techniques and experimenting with hyperparameters.

### Scalability
- **Custom Implementation**: Effective parallel ensemble training using Spark's in-memory processing, resulting in improved efficiency.
- **Spark MLlib**: Demonstrated significant speedup when scaling up the cluster size, especially with the Random Forest Classifier.

## Implementation Details

### Custom Spark + Scikit-Learn Implementation
- **Ensemble Models**: Voting ensemble models and Random Forest classifiers.
- **Key Techniques**: Bagging, custom partitioning, and parallel model execution.

### Spark MLlib Implementation
- **Models Used**: Decision Tree and Random Forest Classifiers.
- **Experiments**: Conducted on AWS EMR clusters, varying cluster sizes and the number of trees in Random Forest.

## Results

### Precision and Recall
- **Custom Implementation**: High precision and recall, especially with Random Forest classifiers.
- **Spark MLlib**: Similar precision and recall values, with notable differences in execution time.

### Running Time
- **Custom Implementation**: Consistently faster execution times, particularly with Decision Tree classifiers.
- **Spark MLlib**: Longer execution times, particularly for larger tree counts in Random Forest classifiers.

## Conclusion

The custom implementation using Spark + Scikit-Learn generally provides faster results, particularly for smaller datasets, while maintaining high accuracy. On the other hand, Spark MLlib offers comparable accuracy but with a trade-off in execution time, particularly for larger datasets or more complex models.

## Repository Structure
- **src/**: Contains the source code for both the custom implementation and Spark MLlib experiments.
- **output/**: Includes output files from the experiments.
- **log/**: Contains log files documenting the execution of the experiments.

## References
- **Dataset**: [Kaggle - Mobile Money Transactions](https://www.kaggle.com/datasets).


---

This README provides a comprehensive overview of the project, including a comparison between the custom implementation and Spark MLlib. Feel free to customize or expand it as needed!
