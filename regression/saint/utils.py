import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, accuracy_score, mean_absolute_error, mean_squared_log_error
import numpy as np
from augmentations import embed_data_mask
import torch.nn as nn
import pandas as pd

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    y_test_upper = torch.empty(0).to(device)
    y_test_lower = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask, image_feature, y_upper, y_lower = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc, image_feature)
            
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_test_upper = torch.cat([y_test_upper,y_upper.squeeze().float()],dim=0)
            y_test_lower = torch.cat([y_test_lower,y_lower.squeeze().float()],dim=0)
            
            #y_pred = torch.cat([y_pred,y_outs],dim=0)
            y_pred = torch.cat([y_pred,y_outs.squeeze().float()],dim=0)
            
        # import ipdb; ipdb.set_trace() 
        
        ## MAE 
        mae = mean_absolute_error(y_test.cpu(), y_pred.cpu())
        ## MSE
        mse = mean_squared_error(y_test.cpu(), y_pred.cpu())
        ## RMSE _ 작을수록 좋음
        #rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        rmsle = np.sqrt(mean_squared_log_error(y_test.cpu(), y_pred.cpu()))
        ## R2 _ 1에 가까운 큰 수일수록 좋음
        r2 = r2_score(y_test.cpu(), y_pred.cpu())
        
        u = y_pred <= y_test_upper #u = torch.le(y_pred, y_test_upper).cpu() # <=
        u = u.cpu()
        l = y_pred >= y_test_lower #l = torch.ge(y_pred, y_test_lower).cpu() # >=
        l = l.cpu()
        compare = pd.concat((pd.DataFrame(u,columns=['upper']), pd.DataFrame(l,columns=['lower'])), axis=1)
        ratio = sum(compare[compare['lower']==True]['upper']==True) / len(compare)
        
        return mae, mse, rmsle, r2, y_test, y_pred, ratio

#y_test.cpu(), y_pred.cpu()