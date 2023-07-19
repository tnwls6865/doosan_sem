import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
import json
import torch
from torchvision import transforms

def data_split(X,y,indices):
    x_d = {
        'data': X.values[indices]
    }        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

def data_prep_openml(feature_num, datasplit=[.65, .15, .2]):
    
    dataset = pd.read_csv(r"/home/jungmin/workspace/doosan/data_all_features_add_image.csv", encoding='UTF-8', sep=',') # integrated version of IN792sx, interrupt, cm939w data

    ## select independent variables
    #X = dataset[['stress_mpa','temp_oc', 'gamma','gammaP','gammaP_aspect','gammaP_width','gammaP_circle']]
    X = dataset[['temp_oc','stress_mpa']]
    #categorical_indicator = [False,  False, False, False, False, False, False]
    categorical_indicator = [False, False]
    #attribute_names = ['stress_mpa','temp_oc','gamma','gammaP','gammaP_aspect','gammaP_width','gammaP_circle']
    attribute_names = ['temp_oc','stress_mpa']

    ## as confidence interval
    y = dataset[['mean']]
    y_upper = dataset[['upper']]
    y_lower = dataset[['lower']]

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    ## train, valid, test split
    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))
    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index
    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    
    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    # MissingValue -> mean
    for col in cont_columns:
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values

    ## dependent variable split
    X_train, y_train = data_split(X,y,train_indices)
    X_valid, y_valid = data_split(X,y,valid_indices)
    X_test, y_test = data_split(X,y,test_indices)

    ## independent variable split
    y_upper_train = {'data': y_upper.values[train_indices]}
    y_upper_valid = {'data': y_upper.values[valid_indices]}
    y_upper_test = {'data': y_upper.values[test_indices]}
    y_lower_train = {'data': y_lower.values[train_indices]}
    y_lower_valid = {'data': y_lower.values[valid_indices]}
    y_lower_test = {'data': y_lower.values[test_indices]}

    ## select image feature
    if feature_num == "1":
        IF = dataset['image_feature_1'].apply(lambda x: torch.FloatTensor(json.loads(x)))
    elif feature_num == "2":
        IF = dataset['image_feature_2'].apply(lambda x: torch.FloatTensor(json.loads(x)))
    elif feature_num == "3":
        IF = dataset['image_feature_3'].apply(lambda x: torch.FloatTensor(json.loads(x)))
    elif feature_num == "4":
        IF = dataset['image_feature_4'].apply(lambda x: torch.FloatTensor(json.loads(x)))
    elif feature_num == "5":
        IF = dataset['image_feature_5'].apply(lambda x: torch.FloatTensor(json.loads(x)))
    elif feature_num == "6":
        IF = dataset['image_feature_6'].apply(lambda x: torch.FloatTensor(json.loads(x)))    
    ## image feature split
    IF_train = {'data': IF.values[train_indices]}
    IF_valid = {'data': IF.values[valid_indices]}
    IF_test = {'data': IF.values[test_indices]}
      
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std) # set infimum of sd to 1e-6
 
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, IF_train, IF_valid, IF_test, y_upper_train, y_upper_valid, y_upper_test, y_lower_train, y_lower_valid, y_lower_test



class DataSetCatCon(Dataset):
    def __init__(self, X, Y, Y_upper, Y_lower, IF, cat_cols,continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64)           #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32)         #numerical columns
        
        self.IF = IF['data']
    
        self.y = Y['data'].astype(np.float32)
        self.y_upper = Y_upper['data'].astype(np.float32)
        self.y_lower = Y_lower['data'].astype(np.float32)
            
        self.cls = np.zeros_like(self.y,dtype=int)

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], self.IF[idx], self.y_upper[idx], self.y_lower[idx] #self.transform(self.IF_train[idx])
