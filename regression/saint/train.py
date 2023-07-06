import torch
from torch import nn
from models import SAINT

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from tqdm import tqdm
import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', default='doosan')
#parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--vision_dset', default="False")
parser.add_argument('--task', default='regression', type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int) ## 32 -> 16
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batchsize', default=216, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 5 , type=int)
#parser.add_argument('--active_log', action = 'store_true')
parser.add_argument('--active_log', type=str)

#parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain', type=str)
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

parser.add_argument('--feature_num', default= "1", choices=["1","2","3","4","5","6"])

opt = parser.parse_args()

modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log=='True':
    import wandb
    if opt.pretrain=='True':
        wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task=='multiclass':
            wandb.init(project="saint_v2_all_kamal", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
   


print('Downloading and processing the dataset, it might take some time.')
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, IF_train, IF_valid, IF_test, y_upper_train, y_upper_valid, y_upper_test, y_lower_train, y_lower_valid, y_lower_test= data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, opt.feature_num, datasplit=[.65, .15, .2])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4,opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32,opt.embedding_size)
    opt.ff_dropout = 0.8

print(nfeat,opt.batchsize)
print(opt)

if opt.active_log=='True':
    wandb.config.update(opt)
train_ds = DataSetCatCon(X_train, y_train, y_upper_train, y_lower_train, IF_train, cat_idxs,opt.dtask,continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, y_upper_valid, y_lower_valid, IF_valid, cat_idxs,opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, y_upper_test, y_lower_test, IF_test, cat_idxs, opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.



model = SAINT(
categories = tuple(cat_dims), 
num_continuous = len(con_idxs),                
dim = opt.embedding_size,                           
dim_out = 1,                       
depth = opt.transformer_depth,                       
heads = opt.attention_heads,                         
attn_dropout = opt.attention_dropout,             
ff_dropout = opt.ff_dropout,                  
mlp_hidden_mults = (4, 2),       
cont_embeddings = opt.cont_embeddings,
attentiontype = opt.attentiontype,
final_mlp_style = opt.final_mlp_style,
y_dim = y_dim
)
vision_dset = opt.vision_dset

if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'

model.to(device)


if opt.pretrain=='True':
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0

best_valid_rmsle = 100000
best_valid_ratio = 0
RMSLE_best_test_rmsle = 100000
ACC_best_test_rmsle = 100000
RMSLE_best_test_mae = 100000
ACC_best_test_mae = 100000
RMSLE_best_test_r2 = -100000
ACC_best_test_r2 = -100000
RMSLE_best_test_ratio = 0
ACC_best_test_ratio = 0



print('Training begins now.')
#for epoch in tqdm(range(opt.epochs)):
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask, image_feature = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device), data[5].to(device)

        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset) 
                  
        reps = model.transformer(x_categ_enc, x_cont_enc, image_feature)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs,y_gts) 
        else:
            loss = criterion(y_outs,y_gts.squeeze()) 
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    # print(running_loss)
    if opt.active_log=='True':
        wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
        'loss': loss.item()
        })
    #if epoch%10==0:
    if epoch%1==0:
            model.eval()
            with torch.no_grad():
                if opt.task in ['binary','multiclass']:
                    accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                    test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                        (epoch + 1, test_accuracy,test_auroc ))
                    if opt.active_log=='True':
                        wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                        wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                    if opt.task =='multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    else:
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                        # if auroc > best_valid_auroc:
                        #     best_valid_auroc = auroc
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy               
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

                else:
                    valid_mae,valid_mse,valid_rmsle,valid_r2, valid_y_test,valid_y_pred, valid_ratio = mean_sq_error(model, validloader, device,vision_dset)    
                    test_mae,test_mse,test_rmsle,test_r2, test_y_test,test_y_pred, test_ratio = mean_sq_error(model, testloader, device,vision_dset)  
 
                    print('[EPOCH %d] VALID | %.3f | RMSE: %.3f / MAE: %.3f / MSE: %.3f / R2: %.3f / ACC: %.3f' %(epoch + 1, valid_ratio, valid_rmsle,valid_mae,valid_mse,valid_r2,valid_ratio ))
                    print('[EPOCH %d]  TEST | %.3f | RMSE: %.3f / MAE: %.3f / MSE: %.3f / R2: %.3f / ACC: %.3f' %(epoch + 1, test_ratio, test_rmsle,test_mae,test_mse,test_r2,test_ratio ))
                    
                    if opt.active_log=='True':
                        wandb.log({'valid_rmsle': valid_rmsle ,'test_rmsle': test_rmsle })     
                    if valid_rmsle < best_valid_rmsle:
                        best_valid_rmsle = valid_rmsle
                        RMSLE_best_test_rmsle = test_rmsle
                        RMSLE_best_test_mae = test_mae
                        RMSLE_best_test_r2 = test_r2
                        RMSLE_best_test_ratio = test_ratio
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    if valid_ratio > best_valid_ratio:
                        best_valid_ratio = valid_ratio
                        ACC_best_test_rmsle = test_rmsle
                        ACC_best_test_mae = test_mae
                        ACC_best_test_r2 = test_r2
                        ACC_best_test_ratio = test_ratio
            model.train()


total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary':
    print('AUROC on best model:  %.3f' %(best_test_auroc))
elif opt.task =='multiclass':
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
else:
    print(f"RMSLE-based || MAE:{RMSLE_best_test_mae} | RMSLE:{RMSLE_best_test_rmsle} | R2:{RMSLE_best_test_r2} | ACC:{RMSLE_best_test_ratio}")
    print(f"ACC-based || MAE:{ACC_best_test_mae} | RMSLE:{ACC_best_test_rmsle} | R2:{ACC_best_test_r2} | ACC:{ACC_best_test_ratio}")
    #print(f'best valid ratio: {best_valid_ratio} | best test ratio: {best_test_ratio} ')
