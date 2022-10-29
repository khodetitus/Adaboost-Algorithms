import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv('titanic_dataset_all_labeled_cleaned.csv')
print(data.head())
data.drop(['Name', 'Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin'], axis=1, inplace=True)
x = data.drop(['Survived'], axis=1)
y = data['Survived']
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=101, stratify=y)
clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=96)
clf.fit(train_x, train_y)
print(clf.score(train_x, train_y))
print(clf.score(test_x, test_y))
