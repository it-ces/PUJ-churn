# Preprocessing model....


# Preprocessing module
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder




def dummies_ohe(df_,cats):
    """
    Returns a dataframe with dummies,and dropped the categorical in original
    the cats arguments receive the cats to transform.
    """
    df = df_.copy()
    df.reset_index(drop=True, inplace=True)
    ohe = OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False)
    dummies = pd.DataFrame(ohe.fit_transform(df[cats]))
    dummies.columns = ohe.get_feature_names_out()  #Names ohe.get_feature_names_out()-> all dummies
    df.drop(columns=cats, inplace=True)
    df = pd.concat([df,dummies], axis=1)
    return df


def std_z(nums, df_, mean, std):
    """
    standardizing nums(numerical) variables
    """
    df = df_.copy()
    binaries = is_binary(df, nums)
    for col in nums:
        if col not in binaries:
            df[col] = (df[col] - mean)/std
    return df


def Xy(df_,target):
    """
    Split the data in X,y to ML implementations
    """
    df = df_.copy()
    X = df.loc[ : , df.columns != target]
    y = df[target]
    return X,y


def is_binary(df_, nums):
    df = df_.copy()
    variables = []
    for var in nums:
        flag = True
        unique = df_[var].unique()
        for value in unique:
            if value not in [0, 1, np.nan, 0.0, 1.0]:
                flag = False
        if flag == True:
            variables.append(var)
    return variables




def breakdown_vars(df):
    """
    This function allow us categorize accodign to numerical or not
    """
    binaries = is_binary(df, df.columns)
    categorial = []
    nonormal = []
    normal = []
    for t in df.columns:
        if (df[t].dtypes=="object" or df[t].dtypes.name=='category') and  t not in binaries:
            categorial.append(t)
        if (df[t].dtypes=="int64" or df[t].dtypes=="float64") and t not in binaries:
                n,p = stats.shapiro(df[t])
                if p<0.05:
                    nonormal.append(t)
                else: 
                    normal.append(t)
    return categorial, binaries, nonormal, normal





# Standardize X_test with information of X_train.



def standardize_X_test(X_train, X_test):
    X_test_ = X_test.copy()
    cats, binaries, nonormal, normal  = breakdown_vars(X_train)
    locations_scales = {}
    for var in normal + nonormal:
        locations_scales[var] = [X_train[var].mean(), X_train[var].std()]
    for var in locations_scales:
        print(var)
        X_test_[var] = (X_test_[var] - locations_scales[var][0])/locations_scales[var][1]
    return X_test_
