# Adaboost-Algorithms
## Implementation of adaboost algorithm with Python

### importing libraries
```python
import pandas as pd
```
### Loading the dataset
+ first reading the data
+ first five rows of the data
```python
data = pd.read_csv('titanic_dataset_all_labeled_cleaned.csv')
print(data.head())
```
![Alt text](https://github.com/khodetitus/Adaboost-Algorithms/blob/main/pictures/Screenshot-from-2021-03-26-07-54-08.png)
### Separating independent and dependent variables
+ independent variables
+ dependent variables
```python
data.drop(['Name', 'Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin'], axis=1, inplace=True)
x = data.drop(['Survived'], axis=1)
y = data['Survived']
```
### Creating the train and test dataset
+ First import libraries
+ divide into train and test sets
```python
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=101, stratify=y)
```
### importing adaboost libraries
```python
from sklearn.ensemble import AdaBoostClassifier
```
### Creating an adaboost instance
```python
clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=96)

```
### Training the model
```python
clf.fit(train_x, train_y)
```
### Calculating score on training data
```python
print(clf.score(train_x, train_y))  #0.8577844311377245
print(clf.score(test_x, test_y))    #0.8251121076233184
```
### Now we have these parameters in AdaBoostClassifier class
+ base_estimator: The model to the ensemble, the default is a decision tree.
+ n_estimators: Number of models to be built.
+ learning_rate: shrinks the contribution of each classifier by this value.
+ random_state: The random number seed, so that the same random numbers generated every time.

## END NOTES
### This was all about the AdaBoost algorithm in this article. Here we saw, how can we ensemble multiple weak learners to get a strong classifier. We also saw the implementation in python of the same.
[More Info](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-adaboost-algorithm-with-python-implementation)