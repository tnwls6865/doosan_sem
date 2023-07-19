import dill
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
        
parser.add_argument('-data_dir', default="gasturbin_data.csv", help="data path")
parser.add_argument('-in792sx_dir', required=True, help="in792sx property feature data path")
parser.add_argument('-in792sx_interrupt_dir', required=True, help="in792sx_interrupt property feature data path")
parser.add_argument('-cm939w_dir', required=True, help="cm939w property feature data path")
parser.add_argument('-save_dir', default="data_all_features_add_image.csv", help="output save directory")

opt = parser.parse_args()


## 1) in792sx 
in792xs = pd.read_csv(opt.in792sx_dir, encoding='UTF-8', sep=',')  #in792sx_dir = r'/HDD/jungmin/doosan/in792sx_features.csv'

# string preprocess
in792xs['id'] = in792xs['Name'].str[7:-14]
in792xs['id'] = in792xs['id'].replace('409','A0409',regex=True)
in792xs['id'] = in792xs['id'].replace('410','A0410',regex=True)
in792xs['id'] = in792xs['id'].replace('411','A0411',regex=True)
in792xs['id'] = in792xs['id'].replace('412','A0412',regex=True)

in792xs = in792xs[['id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]



## 2) in792sx_interrupt
interrupt = pd.read_csv(opt.in792sx_interrupt_dir, encoding='UTF-8', sep=',') #in792sx_interrupt_dir = r'/HDD/jungmin/doosan/interrupt_add/features_in792sx_interrupt.csv'

# string preprocess
interrupt['id'] = interrupt['Name'].str[:4]
interrupt['id'] = interrupt['id'].replace('9_1_','9_1', regex=True)
interrupt['id'] = interrupt['id'].replace('9_2_','9_2', regex=True)
interrupt['id'] = interrupt['id'].replace('9_3_','9_3', regex=True)
interrupt['id'] = interrupt['id'].replace('9_4_','9_4', regex=True)
interrupt['id'] = interrupt['id'].replace('9_5_','9_5', regex=True)
interrupt['id'] = interrupt['id'].replace('7_1_','7_1', regex=True)
interrupt['id'] = interrupt['id'].replace('7_2_','7_2', regex=True)
interrupt['id'] = interrupt['id'].replace('7_3_','7_3', regex=True)
interrupt['id'] = interrupt['id'].replace('7_4_','7_4', regex=True)
interrupt['id'] = interrupt['id'].replace('7_5_','7_5', regex=True)

interrupt = interrupt[['id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]



## 3) cm939w 
cm939w = pd.read_csv(opt.cm939w_dir, encoding='UTF-8', sep=',', header=None) #cm939w_dir = r'/HDD/dataset/doosan/CM939W/features.csv'

# string preprocess
cm939w['id'] = cm939w[1].replace('.png','',regex=True)
cm939w['id'] = cm939w['id'].replace('900_','',regex=True)
cm939w['id'] = cm939w['id'].replace('950_','',regex=True)
cm939w['id'] = cm939w['id'].replace('1000_','',regex=True)
cm939w['id'] = cm939w['id'].replace('c','C',regex=True)
cm939w['id'] = cm939w['id'].replace('_','-',regex=True)

cm939w.columns = ['idx','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle','id']
cm939w = cm939w[['id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]

#----------------------------------------------------------------------------------------------------------------------------------------------

## merge 
in792xs['file'] = 'in792xs'
interrupt['file'] = 'interrupt'
cm939w['file'] = 'cm939w'

in792xs = in792xs[['file','id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]
interrupt = interrupt[['file','id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]
cm939w = cm939w[['file','id','Name','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]

property_features = pd.concat([in792xs, interrupt, cm939w], axis=0)

# Integrated version of original gasutrbin data(w/ in792sx, interrupt, cm939w)
df = pd.read_csv(opt.data_dir, encoding='UTF-8', sep=',') #data_path = r'/home/jungmin/workspace/doosan/doosan_result/gasturbin_data.csv'
df = df[['file','test_id','temp_oc','stress_mpa','LMP','mean','lower','upper']]
df.columns = ['file','id','temp_oc','stress_mpa','LMP','mean','lower','upper']

info_features = pd.merge(df, property_features, on='id')
info_features = info_features[['file_x','id','Name','stress_mpa','temp_oc','LMP','mean','upper','lower','gamma','gammaP','gammaP_distrib','gammaP_aspect','gammaP_width','gammaP_circle']]
info_features.rename(columns={'file_x':'file'})

#----------------------------------------------------------------------------------------------------------------------------------------------

## embedding image features 
## (To use it as an input feature of a transformer)

'''
(dimension)
encoder_feature[0] : B, 1, 448, 640
encoder_feature[1] : B, 64, 224, 320
encoder_feautre[2] : B, 64, 112, 160
encoder_feature[3] : B, 128, 56, 80
encoder_feautre[4] : B, 256, 56, 80
decoder_output  : B, 32, 56, 80
->
image_feature : B, 1, 32
'''

class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=8)
        self.emb = nn.Linear(64,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(217,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(105,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_4(nn.Module):
    def __init__(self):
        super(CNN_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(105,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_5(nn.Module):
    def __init__(self):
        super(CNN_5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(217,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x
    
class CNN_6(nn.Module):
    def __init__(self):
        super(CNN_6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=16, stride=8)
        self.emb = nn.Linear(21,1)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = x.view(1,32,-1)
        x = x.squeeze()
        x = self.emb(x)
        x = x.permute(1,0)
        return x


def feature_process(option, feature_num):
    data = dill.load(open(f'/home/jungmin/workspace/doosan/image_features_{option}_{feature_num}.pkl', 'rb'))

    if feature_num == "f1" :
        cnn = CNN_1()
    elif feature_num == 'f2':
        cnn = CNN_2()
    elif feature_num == 'f3':
        cnn = CNN_3()
    elif feature_num == 'f4':
        cnn = CNN_4()
    elif feature_num == 'f5':
        cnn = CNN_5()
    elif feature_num == 'decoder_output':
        cnn = CNN_6()
    cnn.cuda()
    processed = []
    for i in tqdm(range(0,len(data['feature_map']))):
        processed.append(cnn(data['feature_map'][i]).tolist())
    image_name = data['image_name']
    return processed, image_name    

#----------------------------------------------------------------------------------------------------------------------------------------------
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []
f6 = []

for data_opt in ['in792sx', 'in792sx_interrupt', 'cm939w']:
  f1, image_name = feature_process(data_opt, "f1")
  f2, _ = feature_process(data_opt, "f2")
  f3, _ = feature_process(data_opt, "f3")
  f4, _ = feature_process(data_opt, "f4")
  f5, _ = feature_process(data_opt, "f5")
  f6, _ = feature_process(data_opt, "decoder_output")

  if data_opt == "in792sx":
    image_feature_in792sx = pd.DataFrame([image_name, f1, f2, f3, f4, f5, f6]).T
    image_feature_in792sx.columns=['Name','image_feature_1','image_feature_2','image_feature_3','image_feature_4','image_feature_5','image_feature_6']
  
  elif data_opt == "in792sx_interrupt":
    image_feature_interrupt = pd.DataFrame([image_name, f1, f2, f3, f4, f5, f6]).T
    image_feature_interrupt.columns=['Name','image_feature_1','image_feature_2','image_feature_3','image_feature_4','image_feature_5','image_feature_6']
  
  elif data_opt == "cm939w":
    image_feature_cm939w = pd.DataFrame([image_name, f1, f2, f3, f4, f5, f6]).T
    image_feature_cm939w.columns=['Name','image_feature_1','image_feature_2','image_feature_3','image_feature_4','image_feature_5','image_feature_6']

#----------------------------------------------------------------------------------------------------------------------------------------------

result_interrupt = pd.merge(info_features, image_feature_interrupt, on='Name')
result_in792sx = pd.merge(info_features, image_feature_in792sx, on='Name')
result_cm939w = pd.merge(info_features, image_feature_cm939w, on='Name')

result = pd.concat([result_interrupt, result_in792sx, result_cm939w], axis=0)
result = result.reset_index(drop=True)

result.to_csv(opt.save_dir, index=False,sep=',',encoding='utf-8')

