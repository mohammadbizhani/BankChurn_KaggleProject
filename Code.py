# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:34:52 2024

@author: mooha
"""

# importing librairies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier 
from catboost import CatBoostClassifier
from lazypredict.Supervised import LazyClassifier


# Loading datas
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Make a copy from datas
train_data = train.copy()
test_data = test.copy()

# Check for NaN values
train_data.isnull().sum()
test_data.isnull().sum()

# Info
train_data.info()
test_data.info()

# Describe the datas
train_data.describe()
test_data.describe()


# --------------------------------------------------------------------------------------------------------------------------------- 
# plotting
plt.figure('Countplot Exited per gender')
sns.countplot(data=train_data, hue='Gender', x='Exited', palette='YlGnBu')
plt.title('Exited per Gender')
plt.show()


plt.figure('Countplot Exited per Geography')
sns.countplot(data=train_data, hue='Geography', x='Exited', palette='Dark2')
plt.title('Exited per Geography')
plt.show()


plt.figure('Geography per Exited')
sns.countplot(data=train_data, hue='Exited', y='Geography', palette='coolwarm')
plt.title('Exited per Geography')
plt.show()


plt.figure('Num of products per Exited')
sns.countplot(data=train_data, hue='Exited', x='NumOfProducts', palette='Spectral')
plt.title('Exited per NumOfProducts')
plt.show()


plt.figure('Has credit Card per Exited')
sns.countplot(data=train_data, hue='Exited', x='HasCrCard', palette='YlOrBr')
plt.title('Exited per Has credit Card')
plt.xticks(ticks=[0.0, 1.0], labels=['No','Yes'])
plt.show()


plt.figure('IsActiveMember per Exited')
sns.countplot(data=train_data, hue='Exited', x='IsActiveMember', palette='cubehelix')
plt.title('Exited per Is he/she has active account')
plt.xticks(ticks=[0, 1], labels=['No','Yes'])
plt.show()


plt.figure('Histplot per age')
sns.histplot(data=train_data, x='Age', bins=25, kde=True, color='purple')
plt.title('Histplot Age')
plt.show()


plt.figure('Histplot for balance')
sns.histplot(data=train_data, x='Balance', bins=20, kde=True, color='blue')
plt.title('Histplot Balance')
plt.show()


plt.figure('Histplot for Estimated Salary')
sns.histplot(data=train_data, x='EstimatedSalary', bins=40, kde=True, color='green')
plt.title('Histplot Estimated Salary')
plt.show()


plt.figure('Histplot for Credit Score')
sns.histplot(data=train_data, x='CreditScore', bins=25, kde=True, color='red')
plt.title('Histplot Credit Score')
plt.show()


plt.figure('ScatterPlot for balance')
sns.scatterplot(data=train_data, x='Exited', y='Balance', color='red')
plt.xticks(ticks=[0,1])
plt.title('ScatterPlot for Balance')
plt.show()


plt.figure('Heatmap for numeric datas', figsize=(10,6))
sns.heatmap(train_data.select_dtypes('number').corr(), cmap='viridis', annot=True)
plt.title('Heatmap for numeric datas')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------

# create age category
def categorize_age(age):
    if age >= 18 and age <= 40:
        return 'group1'
    elif age >= 41 and age <= 60:
        return 'group2'
    elif age >= 61 and age <= 80:
        return 'group3'
    else:
        return 'group4'

# Apply the function to the 'Age' column and create a new column named 'Age Category'
train_data['AgeCategory'] = train_data['Age'].apply(categorize_age)
test_data['AgeCategory'] = test_data['Age'].apply(categorize_age)

print (train_data, test_data)


# Remove Id and Surname
train_data.drop(['id','Surname'], axis=1, inplace=True)
test_data.drop(['id','Surname'], axis=1, inplace=True)

# Remove Duplicated rows
train_data.duplicated().sum()
train_data = train_data.drop_duplicates()

# Spliting datas to categorical and numerical features
numeric_features = ['CustomerId','CreditScore','Age','Balance','EstimatedSalary']
categorical_features = ['Geography','Gender','Tenure','NumOfProducts','AgeCategory' ,'HasCrCard','IsActiveMember']

# Making X & Y
X = train_data.drop(['Exited'], axis=1)
Y = train_data['Exited']


# # Standard Scaler for just numeric features
# SS = StandardScaler()
# X[numeric_features] = SS.fit_transform(X[numeric_features])
# test_data[numeric_features] = SS.fit_transform(test_data[numeric_features])


# get dummy
X = pd.get_dummies(X, columns=categorical_features, dtype=int, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_features, dtype=int, drop_first=True)

# Standard Scaler for all of the features(also dummies)
SS = StandardScaler()
X = SS.fit_transform(X)
test_data = SS.fit_transform(test_data)

# split the data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

#-----------------------------------------------------------------------------------------------------------------------------------

# Define the objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 10.0),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_loguniform("gamma", 0.001, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.001, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.001, 1.0),
    }

    # Create the XGBoost classifier
    xgb_model = XGBClassifier(**params, objective='binary:logistic', random_state=1234)

    # Train the model
    xgb_model.fit(X_train, Y_train)

    # Evaluate the performance
    accuracy = accuracy_score(Y_test, xgb_model.predict(X_test))
    
    return accuracy

# Create the optuna study
study = optuna.create_study(direction="maximize")

# Optimize the objective function using Optuna
study.optimize(objective, n_trials=30)

# Get the best parameters
best_params_xgb = study.best_params
best_value_xgb = study.best_value

print("Best parameters:", best_params_xgb)
print("Best value:" , best_value_xgb)


model_xgb = XGBClassifier(**best_params_xgb,objective='binary:logistic',random_state=1234)
model_xgb.fit(X_train, Y_train)
Score_xgb = accuracy_score(Y_test, model_xgb.predict(X_test))
y_pred_prob_xgb = model_xgb.predict_proba(test_data)[:, 1]

output = pd.DataFrame({'id': test.id, 'Exited': y_pred_prob_xgb})
output.to_csv('submission3.csv', index=False)

#--------------------------------------------------------------------------------------------------------------------------------

# XGBoost model (Manual hyperparameters)
xgb = XGBClassifier()

xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)
Score = accuracy_score(Y_test, Y_pred)

print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

#----------------------------------------------------------------------------------------------------------------------------------

# MLP Model (Manual hyperparameters)
mlp = MLPClassifier(random_state=1234, max_iter=1000,batch_size=64, \
                    solver='adam', alpha=0.01, hidden_layer_sizes=(5,40),)
    
mlp.fit(X_train, Y_train)
Y_pred_mlp = mlp.predict(X_test)
Score_mlp = accuracy_score(Y_test, Y_pred_mlp)

print(classification_report(Y_test, Y_pred_mlp))
print(confusion_matrix(Y_test, Y_pred_mlp))

y_pred_prob = mlp.predict_proba(test_data)[:, 1]

output = pd.DataFrame({'id': test.id, 'Exited': y_pred_prob})
output.to_csv('submission.csv', index=False)

#---------------------------------------------------------------------------------------------------------------------------------

# MLP model with optuna
def objective(trial):
    # Define the MLP classifier with parameters to be tuned
    clf = MLPClassifier(random_state=1234, solver='adam',
        hidden_layer_sizes=trial.suggest_int(
            "hidden_layer_sizes",
            10,
            100,
            step=10,
        ),
        activation=trial.suggest_categorical(
            "activation", ["relu", "tanh","logistic", "identity"]
        ),
        alpha=trial.suggest_float("alpha", 0.0001, 0.01),
        learning_rate_init=trial.suggest_float("learning_rate_init", 0.001, 0.1),
        max_iter=trial.suggest_int("max_iter", 100, 300),
    )

    # Train model with selected parameters
    clf.fit(X_train, Y_train)

    # Evaluate model accuracy
    accuracy = accuracy_score(Y_test, clf.predict(X_test))

    return accuracy

# Create Optuna study
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

# Get best parameters
best_parameters_MLP = study.best_params
best_value_MLP = study.best_value

print("Best parameters:", best_parameters_MLP)
print("Best parameters:", best_value_MLP)


MLP_Model = MLPClassifier(hidden_layer_sizes=90,\
                          activation='relu',\
                          alpha=0.00875324390754,\
                          learning_rate_init=0.008884699192394,\
                          max_iter=230,random_state=1234, solver='adam')
    
MLP_Model.fit(X_train, Y_train)
accuracy = accuracy_score(Y_test, MLP_Model.predict(X_test))
y_pred_prob_MLP = MLP_Model.predict_proba(test_data)[:, 1]

output = pd.DataFrame({'id': test.id, 'Exited': y_pred_prob_MLP})
output.to_csv('submission1.csv', index=False)

#----------------------------------------------------------------------------------------------------------------------------------
# lightgbm model with optuna
def objective(trial):
    # Define the LightGBM parameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
        "max_depth": trial.suggest_int("max_depth", 3, 128),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 64),
        "subsample": trial.suggest_uniform("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.2, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }


    # Train a LightGBM model with the given parameters
    model_lgb = LGBMClassifier(**params)
    
    model_lgb.fit(X_train, Y_train)
    
    accuracy = accuracy_score(Y_test, model_lgb.predict(X_test))

    return accuracy

# Create Optuna study
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=30)

# Get best parameters
best_parameters = study.best_params
best_value = study.best_value

# print
print("Best parameters:", best_parameters)
print("Best parameters:", best_value)


Light_model = LGBMClassifier(**best_parameters, random_state=1234)
    
Light_model.fit(X_train, Y_train)
Score_lgb = accuracy_score(Y_test, Light_model.predict(X_test))
y_pred_prob_lgb = Light_model.predict_proba(test_data)[:, 1]

output = pd.DataFrame({'id': test.id, 'Exited': y_pred_prob_lgb})
output.to_csv('submission4.csv', index=False)

#---------------------------------------------------------------------------------------------------------------------------
# Catmodel
cat = CatBoostClassifier(learning_rate=0.07, max_depth=8 , iterations=500, verbose=0, random_state=1234)

cat.fit(X_train, Y_train)
Score_cat = accuracy_score(Y_test, cat.predict(X_test))
y_pred_prob_cat = cat.predict_proba(test_data)[:, 1]

output = pd.DataFrame({'id': test.id, 'Exited': y_pred_prob_cat})
output.to_csv('submission5.csv', index=False)

#---------------------------------------------------------------------------------------------------------------------------
# lazypredict
lazy = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = lazy.fit(X_train, X_test, Y_train, Y_test)

print(models)

#----------------------------------------------------------------------------------------------------------------------------
