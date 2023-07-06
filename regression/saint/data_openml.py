import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
import json
import torch
from torchvision import transforms

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, feature_num, datasplit=[.65, .15, .2]):
    
    np.random.seed(seed) 
    if ds_id == "doosan":
        dataset = pd.read_csv(r"/home/jungmin/workspace/doosan/data_all_features_add_image_0410.csv", encoding='UTF-8', sep=',') #IN792sx, interrupt, cm939w 합친 데이터
       
        #X = dataset[['stress_mpa','temp_oc', 'gamma','gammaP','gammaP_aspect','gammaP_width','gammaP_circle']]
        X = dataset[['temp_oc','stress_mpa']]
        # X = dataset[['gamma','gammaP','gammaP_aspect','gammaP_width','gammaP_circle']]
        #categorical_indicator = [False,  False, False, False, False, False, False]
        categorical_indicator = [False, False]
        #categorical_indicator = [False, False, False, False, False]
        #attribute_names = ['stress_mpa','temp_oc','gamma','gammaP','gammaP_aspect','gammaP_width','gammaP_circle']
        attribute_names = ['temp_oc','stress_mpa']
        #attribute_names = ['gamma','gammaP','gammaP_aspect','gammaP_width','gammaP_circle']

        y = dataset[['mean']]
        y_upper = dataset[['upper']]
        y_lower = dataset[['lower']]
    else:
        dataset = openml.datasets.get_dataset(ds_id)    
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)

    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)
    
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
    #IF = dataset['image_feature'].apply(lambda x: torch.FloatTensor(json.loads(x)))
    
    
    IF_train = {'data': IF.values[train_indices]}#.reshape(-1, 1)} 
    IF_valid = {'data': IF.values[valid_indices]}#.reshape(-1, 1)} 
    IF_test = {'data': IF.values[test_indices]}#.reshape(-1, 1)} 
    
    y_upper_train = {'data': y_upper.values[train_indices]}
    y_upper_valid = {'data': y_upper.values[valid_indices]}
    y_upper_test = {'data': y_upper.values[test_indices]}
    y_lower_train = {'data': y_lower.values[train_indices]}
    y_lower_valid = {'data': y_lower.values[valid_indices]}
    y_lower_test = {'data': y_lower.values[test_indices]}
    
    
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, IF_train, IF_valid, IF_test, y_upper_train, y_upper_valid, y_upper_test, y_lower_train, y_lower_valid, y_lower_test



class DataSetCatCon(Dataset):
    def __init__(self, X, Y, Y_upper, Y_lower, IF, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        self.IF = IF['data']
        
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
            self.y_upper = Y_upper['data'].astype(np.float32)
            self.y_lower = Y_lower['data'].astype(np.float32)
            
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx], self.IF[idx], self.y_upper[idx], self.y_lower[idx] #self.transform(self.IF_train[idx])
