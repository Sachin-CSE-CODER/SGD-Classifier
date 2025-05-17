# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data.
2. Split Dataset into Training and Testing Sets.
3. Train the Model Using Stochastic Gradient Descent (SGD).
4. Make Predictions and Evaluate Accuracy.
5. Generate Confusion Matrix.

## Program:

Program to implement the prediction of iris species using SGD Classifier.

Developed by: SACHIN S

RegisterNumber: 212224040283

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.head()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
Head:

![{11C0464C-1145-4BB8-87EA-487601CCA1D5}](https://github.com/user-attachments/assets/ba35cae1-48c2-4d1d-808b-e7012f1b5423)

Accuracy:

![{024F8632-7891-43E7-8B6E-5E3A032C191A}](https://github.com/user-attachments/assets/ebe7c05b-0005-4c8f-a7e2-244b513c2db1)

Confusion Matrix:

![{4A29AA24-93CD-4CC0-AC28-BBFAEFEC46E3}](https://github.com/user-attachments/assets/270f6b74-3de1-4007-a6be-4895e78c66db)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
