# Library of Geographic Graph Hybrid Network (geographnet)

The python library of geographic graph hybrid network with attention layers (geographnet). 
Current version just supports the PyTorch package of deep geometric learning and 
will extend to the others in the future. This package is for the paper, 
`"Geographic Graph Hybrid Network for Robust Inversion of Particular Matters"` to be published 
in Remote Sensing  

## Major modules

**Model**

* GEOGCon: Function of geographic graph convolution. 
* knnd_geograph: function of knn to support the output of spatial or spatiotemporal weights.
* knngeo: function to retrieve the nearest neighbors with the distance values.  
* GeoGraphPNet: Geographic Graph Hybrid Network for prediction of PM2.5 and PM10. 
            (multilevel geographic graph convolutions plus full residual deep network).
* DataSamplingDSited: Using distance-weighted kdd to sample the data to get the network 
            topology and the corresponding sample data. 
* WNeighborSampler: Function of using distance-weighted kdd to obtain the mini-batch data  
                    to train and test geographic graph hybrid network.

**Helper functions**
* train: Training function of geographic graph hybrid network for PM2.5 and PM10.
* test: Testing function of the trained geographic graph hybrid network for PM2.5 and PM10. 


## Installation
You can directly install it using the following command for the latest version:
```
  pip install geographnet
```

## Note for installation and use 

**Runtime requirements**

geographnet requires installation of PyTorch with support of  PyG (PyTorch Geometric). 
Also Pandas and Numpy should be installed. 

## Use case 
The package provides one example for use of geographic graph hybrid network to predict PM2.5 and PM10 in mainland China.
See the following example. 

## License

The geographnet is provided under a MIT license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Test call

**Load the packages**
```python
import pandas as pd
import numpy as np
import torch
import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import urllib
import tarfile

from geographnet.model.wdatasampling import DataSamplingDSited
from geographnet.model.wsampler import WNeighborSampler
from geographnet.traintest_pm import train, test
from geographnet.model.geographpnet import GeoGraphPNet
```

**Helper function for site-based independent test and downloading 
```python
def selectSites(datain):
    sitesDF = datain.drop_duplicates('id').copy()
    sgrp = sitesDF['stratified_flag'].value_counts()
    sitesDF['stratified_flag_cnt'] = sgrp.loc[sitesDF['stratified_flag']].values
    pos1_index = np.where(sitesDF['stratified_flag_cnt'] < 5)[0]
    posT_index = np.where(sitesDF['stratified_flag_cnt'] >= 5)[0]
    np.random.seed()
    trainsiteIndex, testsiteIndex = train_test_split(posT_index, stratify=sitesDF.iloc[posT_index]['stratified_flag'],
                                                     test_size=0.15)
    selsites = sitesDF.iloc[testsiteIndex]['id']
    trainsitesIndex = np.where(~datain['id'].isin(selsites))[0]
    indTestsitesIndex = np.where(datain['id'].isin(selsites))[0]
    return trainsitesIndex,indTestsitesIndex

def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs)
```
**Download and extract the sample dataset**

```python
url = 'https://github.com/lspatial/geographnet/raw/master/pmdatain.pkl.tar.gz'
tarfl='./pmdatain.pkl.tar.gz'
print("Downloading from "+url+' ... ...')
urllib.request.urlretrieve(url, tarfl)
target='.'
untar(tarfl,target)
```

**Read and preprocess the data** 
```python
targetFl=target+'/pmdatain.pkl'
datatar=pd.read_pickle(targetFl)
print(datatar.columns,datatar.shape)
covs=['idate','lat', 'lon', 'latlon', 'DOY', 'dem', 'OVP10_TOTEXTTAU', 'OVP14_TOTEXTTAU',
       'TOTEXTTAU', 'glnaswind', 'maiacaod', 'o3', 'pblh', 'prs', 'rhu', 'tem',
       'win', 'GAE', 'NO2_BOT', 'NO_BOT', 'PM25_BOT', 'PM_BOT', 'OVP10_CO',
       'OVP10_GOCART_SO2_VMR', 'OVP10_NO', 'OVP10_NO2', 'OVP10_O3', 'BCSMASS',
       'DMSSMASS', 'DUSMASS25', 'HNO3SMASS', 'NISMASS25', 'OCSMASS', 'PM25',
       'SO2SMASS', 'SSSMASS25', 'sdist_roads', 'sdist_poi', 'parea10km',
       'rlen10km', 'wstag', 'wmix', 'CLOUD', 'MYD13C1.NDVI',
       'MYD13C1.EVI', 'MOD13C1.NDVI', 'MOD13C1.EVI', 'is_workday', 'OMI-NO2']
target=['PM10_24h', 'PM2.5_24h']
X = datatar[covs].values
scX = preprocessing.StandardScaler().fit(X)
Xn = scX.transform(X)
y = datatar[['pm25_log','pm10_log']].values
ypm25 = datatar['PM2.5_24h'].values
ypm10 = datatar['PM10_24h'].values
scy = preprocessing.StandardScaler().fit(y)
yn = scy.transform(y)
```

**Sampling**
```python
tarcols=[i for i in range(len(covs))]
trainsitesIndex=[i for i in range(datatar.shape[0])]
trainsitesIndex, indTestsitesIndex=selectSites(datatar)
x, edge_index,edge_dist, y, train_index, test_index = DataSamplingDSited(Xn[:,tarcols], yn, [0,1,2], 12,
                        trainsitesIndex ,datatar)
Xn = Xn[:, 1:]
edge_weight=1.0/(edge_dist+0.00001)
neighbors=[12,12,12,12]
train_loader = WNeighborSampler(edge_index, edge_weight= edge_weight,node_idx=train_index,
                               sizes=neighbors, batch_size=2048, shuffle=True,
                               num_workers=20 )
x_index = torch.LongTensor([i for i in range(Xn.shape[0])])
x_loader = WNeighborSampler(edge_index, edge_weight= edge_weight,node_idx=x_index,
                           sizes=neighbors, batch_size=2048, shuffle=False,
                           num_workers=20 )
```
**Definition of the model and parameters**
```python
gpu=0
if gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+str(gpu))
nout=2
resnodes = [512, 320, 256, 128, 96, 64, 32, 16]
# 0: original; 1: concated ; 2: dense; 3: only gcn
gcnnhiddens = [128,64,32]
model = GeoGraphPNet(x.shape[1], gcnnhiddens, nout, len(neighbors), resnodes, weightedmean=True,gcnout=nout,nattlayer=1)
model = model.to(device)
x = x.to(device)
edge_index = edge_index.to(device)
y = y.to(device)
init_lr=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
best_indtest_r2 = -9999
best_indtest_r2_pm10=-9999
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
oldlr=newlr=init_lr
```

**Model training**
```python
epoch=0
nepoch=5
trpath="./test"
while epoch< nepoch  :
    # adjust_lr(optimizer, epoch, init_lr)
    print('Conducting ',epoch, ' of ',nepoch,' for PM  ... ...')
    loss,loss_pm25,loss_pm10,loss_rel = train(model, train_loader, device, optimizer, x, y)
    try:
       permetrics,lossinf,testdata= test(model, x_loader, device, x, y, scy,train_index,
                         test_index, indtest_index=indTestsitesIndex,
                         ypm25=ypm25 ,ypm10=ypm10)
       lossall, lossall_pm25, lossall_pm10, lossall_rel = lossinf
       pmindtesting, pmtesting, pmtrain=testdata
    except:
        print("Wrong loop for ecpoch "+str(epoch)+ ", continue ... ...")
        epoch=epoch+1
        continue
    permetrics_pm25 = permetrics[permetrics['pol'] == 'pm2.5']
    permetrics_pm10 = permetrics[permetrics['pol'] == 'pm10']
    permetrics_pm25=permetrics_pm25.iloc[0]
    permetrics_pm10 = permetrics_pm10.iloc[0]
    if epoch>15 and permetrics_pm25['train_r2']<0 :
        print("Abnormal for ecpoch " + str(epoch) + ", continue ... ...")
        epoch = epoch + 1
        continue
    if best_indtest_r2 < permetrics_pm25['indtest_r2']:
        best_indtest_r2 = permetrics_pm25['indtest_r2']
        saveDf = pd.DataFrame({'sid': datatar.iloc[test_index]['sid'].values, 'obs': pmtesting['pm25_obs'].values,
                               'pre': pmtesting['pm25_pre'].values})
        saveindDf = pd.DataFrame({'sid': datatar.iloc[indTestsitesIndex]['sid'].values, 'obs': pmindtesting['pm25_obs'].values,
             'pre': pmindtesting['pm25_pre'].values})
        testfl = trpath + '/model_pm25_bestindtest_testdata.csv'
        saveDf.to_csv(testfl,index_label='index')
        indtestfl = trpath + '/model_pm25_bestindtest_indtestdata.csv'
        saveindDf.to_csv(indtestfl, index_label='index')
        modelFl = trpath + '/model_pm25_bestindtestr2.tor'
        torch.save(model, modelFl)
        modelMeFl = trpath + '/model_pm25_bestindtestr2.csv'
        pd.DataFrame([permetrics_pm25.to_dict()]).to_csv(modelMeFl, index_label='epoch')

    if best_indtest_r2_pm10 < permetrics_pm10['indtest_r2']:
        best_indtest_r2_pm10 = permetrics_pm10['indtest_r2']
        saveDf = pd.DataFrame({'sid': datatar.iloc[test_index]['sid'].values, 'obs': pmtesting['pm10_obs'].values,
                               'pre': pmtesting['pm10_pre'].values})
        saveindDf = pd.DataFrame(
            {'sid': datatar.iloc[indTestsitesIndex]['sid'].values, 'obs': pmindtesting['pm10_obs'].values,
             'pre': pmindtesting['pm10_pre'].values})
        testfl = trpath + '/model_pm10_bestindtest_testdata.csv'
        saveDf.to_csv(testfl, index_label='index')
        indtestfl = trpath + '/model_pm10s_bestindtest_indtestdata.csv'
        saveindDf.to_csv(indtestfl, index_label='index')
        modelFl = trpath + '/model_pm10_bestindtestr2.tor'
        torch.save(model, modelFl)
        modelMeFl = trpath + '/model_pm10_bestindtestr2.csv'
        pd.DataFrame([permetrics_pm10.to_dict()]).to_csv(modelMeFl, index_label='epoch')
    scheduler.step(loss)
    newlr= optimizer.param_groups[0]['lr']
    if newlr!=oldlr:
        print('Learning rate is {} from {} '.format(newlr, oldlr))
        oldlr=newlr
    atrainDf=permetrics
    atrainDf['epoch']=epoch
    lossDf=pd.DataFrame({'epoch':epoch,'loss':loss, 'loss_pm25':loss_pm25,'loss_pm10':loss_pm10,
                         'loss_rel':loss_rel,'lossall':lossall,'lossall_pm25':lossall_pm25,
                         'lossall_pm10':lossall_pm10,'lossall_rel':lossall_rel},index=[epoch])
    print(permetrics)
    print(lossDf)
    if epoch==0:
        alltrainHist=atrainDf
        alllostinfo=lossDf
    else:
        alltrainHist=alltrainHist.append(atrainDf)
        alllostinfo = alllostinfo.append(lossDf)
    epoch=epoch+1
```

**Save the results**
```python
tfl = trpath + '/trainHist.csv'
alltrainHist.to_csv(tfl, header=True, index_label="row")
tfl = trpath + '/ftrain_loss.csv'
alllostinfo.to_csv(tfl, header=True, index_label="row")
del optimizer, x, edge_index, y, train_index, test_index, model, alltrainHist
gc.collect()
```
## Collaboration

Welcome to contact Dr. Lianfa Li (Email: lspatial@gmail.com). 
