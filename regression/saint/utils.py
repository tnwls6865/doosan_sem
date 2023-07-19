import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
import numpy as np
import pandas as pd


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  


def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler


def embed_data(x_categ, x_cont, model):
    device = x_cont.device
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    
    #image_feature_enc = model.embeds(image_feature)
    x_cont_enc = x_cont_enc.to(device)        
    return x_categ, x_categ_enc, x_cont_enc#, image_feature_enc


def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    y_test_upper = torch.empty(0).to(device)
    y_test_lower = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask, image_feature, y_upper, y_lower = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
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
