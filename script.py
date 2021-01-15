import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

import joblib

# Data
def pred():

    data = pd.read_csv('./tables/train.csv')
    test_data = pd.read_csv('./tables/test_no_target1.csv')

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(data, data['price'],
                                                        test_size=0.1,
                                                        random_state=0)

    print(X_train.shape, X_test.shape)

    # Missing values
    vars_with_na = [
        var for var in data.columns
        if X_train[var].isnull().sum() > 0 and X_train[var].dtypes == 'O'

        if test_data[var].isnull().sum() > 0 and test_data[var].dtypes == 'O'
    ]

    print(test_data[vars_with_na].isnull().mean())

    X_train[vars_with_na] = X_train[vars_with_na].fillna('Missing')
    X_test[vars_with_na] = X_test[vars_with_na].fillna('Missing')

    test_data[vars_with_na] = test_data[vars_with_na].fillna('Missing')
    test_data.head()

    # Year
    def change_years(df):
        df = df.copy()
        for var in df['registration_year']:
            if var < 20:
                df['registration_year'] = df['registration_year'].replace(var, var + 2000)
            elif var < 100:
                df['registration_year'] = df['registration_year'].replace(var, var + 1900)

        return df


    X_train = change_years(X_train)
    test_data = change_years(test_data)

    # Numerical values
    vars_with_na = [
        var for var in data.columns
        if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'O'

        if test_data[var].isnull().sum() > 0 and test_data[var].dtypes != 'O'
    ]

    X_train[vars_with_na].isnull().mean()

    print(vars_with_na)

    for var in vars_with_na:

        mode_val = X_train[var].mode()[0]
        mode_val_test = test_data[var].mode()[0]

        X_train[var + '_na'] = np.where(X_train[var].isnull(), 1, 0)
        X_test[var + '_na']=np.where(X_test[var].isnull(), 1, 0)

        test_data[var + '_na']=np.where(test_data[var].isnull(), 1, 0)

        X_train[var] = X_train[var].fillna(mode_val)
        X_test[var] = X_test[var].fillna(mode_val)

        test_data[var] = test_data[var].fillna(mode_val_test)


    print(X_train[vars_with_na].isnull().sum())

    # Variable transformation
    def change_power(train, test):
        df = train.copy()
        df1 = test.copy()

        df['power'] = df['power'].replace(0, np.nan)
        mode = df['power'].mode(True)
        df['power'] = df['power'].replace(np.nan, int(mode))

        df1['power'] = df1['power'].replace(0, np.nan)
        mode1 = df1['power'].mode(True)
        df1['power'] = df1['power'].replace(np.nan, int(mode1))

        return df, df1


    X_train, X_test = change_power(X_train, X_test)

    test_data, X_test = change_power(test_data, X_test)

    def change_capacity(train, test):
        df = train.copy()
        df1 = test.copy()

        df['engine_capacity'] = df['engine_capacity'].replace(0, np.nan)
        mode = df['engine_capacity'].mode(True)
        print(int(mode))
        df['engine_capacity'] = df['engine_capacity'].replace(np.nan, int(mode))

        df1['engine_capacity'] = df1['engine_capacity'].replace(0, np.nan)
        mode1 = df1['engine_capacity'].mode(True)
        df1['engine_capacity'] = df1['engine_capacity'].replace(np.nan, int(mode1))

        return df, df1


    X_train, X_test = change_capacity(X_train, X_test)

    test_data, X_test = change_capacity(test_data, X_test)


    for var in ['engine_capacity', 'insurance_price', 'price', 'power', 'mileage']:
        X_train[var] = np.log(X_train[var])
        X_test[var] = np.log(X_test[var])

    #
    for var in ['engine_capacity', 'insurance_price', 'power', 'mileage']:
        test_data[var] = np.log(test_data[var])



    # Categorial variables
    cat_vars = [var for var in X_train.columns if X_train[var].dtype == 'O']
    def find_frequent_labels(df, var, rare_perc):

        df = df.copy()
        tmp = df.groupby(var)['price'].count() / len(df)
        return tmp[tmp > rare_perc].index

    for var in cat_vars:
        frequent_ls = find_frequent_labels(X_train, var, 0.01)
        X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
        X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')
        test_data[var] = np.where(test_data[var].isin(frequent_ls), test_data[var], 'Rare')


    def replace_categories(train, test, var, target, test_data):

        ordered_labels = train.groupby([var])[target].mean().sort_values().index
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
        train[var] = train[var].map(ordinal_label)
        test[var] = test[var].map(ordinal_label)

        test_data[var] = test_data[var].map(ordinal_label)


    for var in cat_vars:
        replace_categories(X_train, X_test, var, 'price', test_data)

    print(test_data.head())


    # Feature scaling
    train_vars = [var for var in X_train.columns if var not in ['Id', 'price', 'zipcode']]

    len(train_vars)
    X_train.shape
    X_train.head(20)
    np.any(np.isnan(X_test))
    np.all(np.isfinite(X_test))
    # X_train = X_train.values.astype(np.float)
    X_train.head()
    # y = y.values.astype(np.float)
    scaler = MinMaxScaler()

    scaler.fit(X_train[train_vars])

    X_train[train_vars] = scaler.transform(X_train[train_vars])
    X_test[train_vars] = scaler.transform(X_test[train_vars])

    test_data[train_vars] = scaler.transform(test_data[train_vars])

    X_train.to_csv('./tables/xtrain.csv', index=False)
    X_test.to_csv('./tables/xtest.csv', index=False)

    test_data.to_csv('./tables/test_data.csv', index=False)


    # Step 2
    import xgboost
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer, r2_score

    X_train = pd.read_csv('./tables/xtrain.csv')
    X_test = pd.read_csv('./tables/xtest.csv')

    y_train = X_train['price']
    y_test = X_test['price']

    features = ['type',
     'registration_year',
     'gearbox',
     'power',
     'model',
     'mileage',
     'fuel',
     'brand',
     'damage',
     'insurance_price']

    X_train = X_train[features]
    X_test = X_test[features]



    def test_model(model, X_train=X_train, y_train=y_train):
        cv = KFold(n_splits=3, shuffle=True, random_state=45)
        r2 = make_scorer(r2_score)
        r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=r2)
        score = [r2_val_score.mean()]
        return score

    xgb2_reg=xgboost.XGBRegressor(n_estimators=899,
                                 mon_child_weiht=2,
                                 max_depth=4,
                                 learning_rate=0.05,
                                 booster='gbtree')

    test_model(xgb2_reg)

    test_data = pd.read_csv('./tables/test_data.csv')
    test_data_id = pd.read_csv('./tables/test_data.csv')
    test_data = test_data[features]
    test_data_id.head()

    xgb2_reg.fit(X_train, y_train)
    y_pred = np.exp(xgb2_reg.predict(test_data))
    submit_test = pd.concat([test_data_id['Id'], pd.DataFrame(y_pred)], axis=1)
    submit_test.columns=['Id', 'Predicted']
    submit_test.to_csv('./tables/submission1.csv', index=False)
    submit_test

    model_path = 'models/'
    model_file = f'v1-{datetime.now()}.model'.replace(' ', '_')

    joblib.dump(xgb2_reg, model_path + model_file)

    return y_pred







pred()
