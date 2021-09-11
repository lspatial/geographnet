import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import r2_score as r2
import numpy as np
import torch
import pandas as pd

def rmse(predictions, targets):
    """ RMSE function  
       Args:
           predictions (1-d array of float): Predicted values.
           targets (1-d array of float): Observed values.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def train(model,dataloader,device,optimizer,X,y):
    """ training function for geographic graph hybrid network
       Args:
           model (GeoGraphPNet): Model to be trained.
           dataloader (WNeighborSampler): Mini-batch sampler for the model.
           device (int): GPU index.
           optimizer (optim): specific optimizer like Adam.
           X (2-d array): Input data.
           y (1-d array): Observed values of the dependent (target) variable.
    """
    model.train()
    total_loss = 0
    total_loss_pm25 = 0
    total_loss_pm10 = 0
    total_loss_rel = 0
    ypre_b,yobs_b=[],[]
    for batch_size, n_id, adjs in tqdm(dataloader):
        if not isinstance(adjs,list):
            adjs=[adjs]
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(X[n_id], adjs, X[n_id[:batch_size]])   # .reshape(-1)
        pm25=out[:,0]
        pm10=out[:,1]
        dif=out[:,2]
        grd=y[n_id[:batch_size]]    #.reshape(-1)
        pm25o=grd[:,0]
        pm10o=grd[:,1]
        nm=torch.sum(torch.square(dif))
        loss1 = F.mse_loss(pm25,pm25o)
        loss2 = F.mse_loss(pm10,pm10o)
        loss3= nm
        loss = loss1+loss2+loss3*0.6
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_loss_pm25+=float(loss1)
        total_loss_pm10 += float(loss2)
        total_loss_rel += float(loss3)
        ypre_b.append(out.cpu().detach().numpy())
        yobs_b.append(grd.cpu().detach().numpy())
    loss = total_loss / X.shape[0]
    loss_pm25 = total_loss_pm25 / X.shape[0]
    loss_pm10 = total_loss_pm10 / X.shape[0]
    loss_rel = total_loss_rel / X.shape[0]
    return loss,loss_pm25,loss_pm10,loss_rel

@torch.no_grad()
def test(model, dataloader,device,X,y,scy,train_index,test_index, indtest_index=None,ypm25=None,ypm10=None):
    r""" test function for geographic graph hybrid network
       Args:
           model (GeoGraphPNet): Trained model. 
           dataloader (WNeighborSampler): Mini-batch sampler for the model.
           device (int): GPU index.
           X (2-d array): Input data.
           y (1-d array): Observed values of the dependent (target) variable.
           scy (StandardScaler): StandardScaler object used to transform the predicted values into the original scale.
           train_index (int list or array): index of training samples.
           test_index (int list or array): index of testing samples.
           indtest_index (int list or array): index of site-based independent testing samples.
           ypm25 (1-d array): Observed values of PM2.5 concentration.
           ypm10 (1-d array): Observed values of PM10 concentration.
    """
    model.eval()
    total_loss = 0
    total_loss_pm25 = 0
    total_loss_pm10 = 0
    total_loss_rel = 0
    ypre_b,yobs_b=[],[]
    for batch_size, n_id, adjs in tqdm(dataloader):
        if not isinstance(adjs,list):
            adjs=[adjs]
        adjs = [adj.to(device) for adj in adjs]
        out = model(X[n_id], adjs,X[n_id[:batch_size]])
        grd=y[n_id[:batch_size]]
        pm25 = out[:, 0]
        pm10 = out[:, 1]
        dif = out[:, 2]
        grd = y[n_id[:batch_size]]  # .reshape(-1)
        pm25o = grd[:, 0]
        pm10o = grd[:, 1]
        nm = torch.mean(torch.sqrt(dif))
        loss1 = F.mse_loss(pm25,pm25o)
        loss2 = F.mse_loss(pm10,pm10o)
        loss3= nm
        loss = loss1+loss2+loss3*0.6
        total_loss += float(loss)
        total_loss_pm25 += float(loss1)
        total_loss_pm10 += float(loss2)
        total_loss_rel += float(loss3)
        ypre_b.append(out.cpu().detach().numpy())
        yobs_b.append(grd.cpu().detach().numpy())
    loss = total_loss / X.shape[0]
    loss_pm25 = total_loss_pm25 / X.shape[0]
    loss_pm10 = total_loss_pm10 / X.shape[0]
    loss_rel = total_loss_rel / X.shape[0]
    ypre=np.concatenate(ypre_b)
    pm=scy.inverse_transform(ypre[:, [0,1]])
    ypre_pm25 = np.clip(pm[:,0], a_min=None, a_max=9.5)
    ypre_pm10 = np.clip(pm[:,1], a_min=None, a_max=9.5)
    ypre_pm25=np.exp(ypre_pm25)
    ypre_pm10=np.exp(ypre_pm10)
    ypre_pm25 = np.array(ypre_pm25, dtype=np.float64)
    ypre_pm10 = np.array(ypre_pm10, dtype=np.float64)
    trainpm25_r2 = r2(ypm25[train_index], ypre_pm25[train_index])
    trainpm25_rmse = rmse(ypm25[train_index], ypre_pm25[train_index])
    trainpm10_r2 = r2(ypm10[train_index], ypre_pm10[train_index])
    trainpm10_rmse = rmse(ypm10[train_index], ypre_pm10[train_index])

    testpm25_r2 = r2(ypm25[test_index], ypre_pm25[test_index])
    testpm25_rmse = rmse(ypm25[test_index], ypre_pm25[test_index])
    testpm10_r2 = r2(ypm10[test_index], ypre_pm10[test_index])
    testpm10_rmse = rmse(ypm10[test_index], ypre_pm10[test_index])

    indtestpm25_r2 = r2(ypm25[indtest_index], ypre_pm25[indtest_index])
    indtest25_rmse = rmse(ypm25[indtest_index], ypre_pm25[indtest_index])
    indtestpm10_r2 = r2(ypm10[indtest_index], ypre_pm10[indtest_index])
    indtestpm10_rmse = rmse(ypm10[indtest_index], ypre_pm10[indtest_index])

    pmindtesting=pd.DataFrame({'pm10_obs':ypm10[indtest_index], 'pm10_pre':ypre_pm10[indtest_index],
                              'pm25_obs':ypm25[indtest_index], 'pm25_pre':ypre_pm25[indtest_index]},
                             index=range(len(indtest_index)))
    pmtesting=pd.DataFrame({'pm10_obs':ypm10[test_index], 'pm10_pre':ypre_pm10[test_index],
                              'pm25_obs':ypm25[test_index], 'pm25_pre':ypre_pm25[test_index]},
                             index=range(len(test_index)))
    pmtrain=pd.DataFrame({'pm10_obs':ypm10[train_index], 'pm10_pre':ypre_pm10[train_index],
                              'pm25_obs':ypm25[train_index], 'pm25_pre':ypre_pm25[train_index]},
                             index=range(len(train_index)))
    permetrics=pd.DataFrame({'pol':['pm2.5','pm10'],'train_r2':[trainpm25_r2,trainpm10_r2],'train_rmse':[trainpm25_rmse,trainpm10_rmse],
                  'test_r2':[testpm25_r2,testpm10_r2],'test_rmse':[testpm25_rmse,testpm10_rmse],
                  'indtest_r2':[indtestpm25_r2,indtestpm10_r2],'indtest_rmse':[indtest25_rmse,indtestpm10_rmse]},index=['pm2.5','pm10'])
    return permetrics,(loss,loss_pm25,loss_pm10,loss_rel),(pmindtesting,pmtesting,pmtrain)
