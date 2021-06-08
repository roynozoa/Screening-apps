# Screening-apps
Aplikasi yang dapat menentukan apakah seseorang boleh mendapatkan vaksin COVID-19 atau belum berdasarkan data riwayat penyakit.

Mata Kuliah Manajemen Proyek TI - 2021
Oleh:
- Stefannov
- M. Adisatriyo Pratama
- Surya Asmoro

## Deskripsi Projek
- Aplikasi yang dapat menentukan apakah seseorang
boleh mendapatkan vaksin COVID-19 atau belum
berdasarkan data riwayat penyakit.
- Aplikasi berbentuk web yang dapat diakses melalui
internet.
- Aplikasi ini menerapkan metode Machine Learning


## Team Members and Roles
- Muhammad Adisatriyo P :
    - Project Management
    - Build ML model
- Surya Asmoro:
    - Gather Data
    - Cleaning Data
    - Software Testing
- Stefannov:
    - Gather Data
    - Cleaning Data
    - Software Testing


## Background Problem

## Tools
- Python3
- Jupyter Notebook and Google Colab
- Scikit-Learn
- Google Docs
- Github
- Streamlit Framework
- Google Cloud Platform

# Machine Learning Model

Our problem is really simple binary Classification with just using decision tree classifier as our main algorithm and several parameters that is provided from scikit-learn can easily reach up to ~99% test accuracy.

### here is the sample code

#### first let's split our dataset into training and testing

```
from sklearn.model_selection import train_test_split

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

```
#### Initialize model algorithm

```
# import library
from sklearn.tree import DecisionTreeClassifier

# model initialize with max_depth = 2
dt = DecisionTreeClassifier(max_depth=2, random_state=42)
```

#### with that algorithm can have up to ~90% accuracy
```
from sklearn.metrics import accuracy_score

# fit training set
dt.fit(X_train, y_train)

# predict test set
y_pred = dt.predict(X_test)

# accuracy score
accuracy_score(y_test, y_pred)
```
### output
```
0.9180327868852459
```

#### using GridSearchCV for searching best hyperparameter
```
# Import library
from sklearn.model_selection import GridSearchCV

## hyperparameter
params_dt = {'max_depth':[1,2,3,4,6,8,10], 'min_samples_leaf':[0.0001, 0.001, 0.05, 0.1, 0.2], 'criterion':['gini', 'entropy']}

# performs GridSearchCV
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=1)

# fit data
grid_dt.fit(X_train, y_train)

print(f'Best Parameters : {grid_dt.best_params_}')
print(f'Best Score : {grid_dt.best_score_}')
print(f'Best Estimator : {grid_dt.best_estimator_}')
```

#### The best score is up to ~99% accuracy
#### output
```
Best Parameters : {'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 0.0001}
Best Score : 0.9989732930209121
Best Estimator : DecisionTreeClassifier(max_depth=8, min_samples_leaf=0.0001, random_state=42)
```

#### Saving the best model using pickle
```
import pickle
pickle.dump(grid_dt.best_estimator_, open('clf.pkl', 'wb'))
```

## Deployment

We are using Google Cloud Platform to quickly deploy our web app. We are using Google App Engine and fetch the data from GitHub repository to build the web app.

Here is our web app URL: http://www.screening-apps.info/


## Reference

- Python Docs: https://docs.python.org/3/
- Google Cloud Platform: https://cloud.google.com/
- Streamlit: https://streamlit.io/
- Scikit-learn: https://scikit-learn.org/