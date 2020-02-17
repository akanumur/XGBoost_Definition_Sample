# XGBoost_Definition_Sample
Simple Implementation of XGBoost on Boston Housing Dataset

## What is XGBoosting and what are the advantages of XGBoost?
XGBoost belongs to the family Boosting algorithms. It is a method of sequential boosting, where the input for current classifier is from the previous classifier. The values which are predicted accurately are given less weights compared to the ones which are misclassified. By doing this the ML reduces the misclassification error and increases the prediction rate.That's the basic idea behind boosting algorithms is building a weak model, making conclusions about the various feature importance and parameters, and then using those conclusions to build a new, stronger model and capitalize on the misclassification error of the previous model and try to reduce it.

## Advantages of XGBoost:
- Speed and performance : Originally written in C++, it is comparatively faster than other ensemble classifiers.

- Core algorithm is parallelizable : Because the core XGBoost algorithm is parallelizable it can harness the power of multi-core computers. It is also parallelizable onto GPU’s and across networks of computers making it feasible to train on very large datasets as well.

- Consistently outperforms other algorithm methods : It has shown better performance on a variety of machine learning benchmark datasets.

- Wide variety of tuning parameters : XGBoost internally has parameters for cross-validation, regularization, user-defined objective functions, missing values, tree parameters, scikit-learn compatible API etc.

## XGBoost on Boston Housing dataset
### About the data
The dataset for this project originates from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts.

Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:  

    :Number of Instances: 506

    :Number of Attributes: 13 numeric/categorical predictive

    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   

**References**

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
   
Before implementing any Ensemble Algorithms it is better to convert categorical features to one-hot encoded variables which means if a variables has several levels. For Example Pet variable has different levels as CAT, DOG and RABBIT. It is better to one-hot encode because the ML may be influenced by the variable which has more levels. So we can convert our PET variable as follows IS_CAT, IS_DOG, IS_RABBIT the output would be a binary output if yes 1 or 0.
XGBoost has ability to handle missing values so it is okay if we don't replace with NA.

After creating the dataset, we need to convert the dataset to a Dmatrix.**DMatrix** is a internal data structure that used by XGBoost which is optimized for both memory efficiency and training speed. You can construct DMatrix from numpy.arrays Parameters.

## XGBoost Hyperparameters:

XGBoost provides a wide range of tuning parameters:
- `learning_rate`: step size shrinkage used to prevent overfitting. Range is [0,1]
- `max_depth`: determines how deeply each tree is allowed to grow during any boosting round.
- `subsample`: percentage of samples used per tree. Low value can lead to underfitting.
- `colsample_bytree`: percentage of features used per tree. High value can lead to overfitting.
- `n_estimators`: number of trees you want to build.
- `objective`: determines the loss function to be used like `reg:linear` for regression problems, `reg:logistic` for classification problems with only decision, `binary:logistic` for classification problems with probability.

XGBoost also provides various regularization parameters
- `gamma`: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.
- `alpha`: L1 regularization on leaf weights. A large value leads to more regularization.
- `lambda`: L2 regularization on leaf weights and is smoother than L1 regularization.

For a regression problem we use `XGBRegressor()` for a classifier problem we use `XGBClassifier()`.
As the current business problem is to predict the housing price we use RSME as the evaluation metric. The lower RMSE value the better is the model.

We can also use cross validation method to improve our model in XGBoost as XGBoost supports cross validation through `cv()` method.
Illustration of xgboost() in python is attached in the code.
