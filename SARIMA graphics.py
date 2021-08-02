'''SARIMA'''
import numpy as np
import textwrap as tw
import pandas as pd #for reading quickly the data files
import os #to get the current working directory
from sklearn import preprocessing as skp
import matplotlib #for plotting the results of the model
#matplotlib.use('Agg')  #to create image files without showing them
import matplotlib.pyplot as plt #to add values and organize the image
#%matplotlib inline

#finalloc='D:/ThesisExperiments/OriginalData/ResultsRNN24'
#finalloc='D:/ThesisExperiments/OriginalData/ResultsB/ResultsRNN24'
finalloc='D:/ThesisExperiments/OriginalData/ResultsSarima/'
#finalloc='D:/ThesisExperiments/OriginalData/ResultsB/BestResultsRNN24'
#len(sorted(os.listdir(path=finalloc+"Loses/GR9")))

'''train and test'''

# Hour=[24,12,1]
# Lay=[2,5,8]
# NodnoCEA=[3,13,23]
# NodsiCEA=[11,21,31]
# Nods=[14,24,34]
# AR=[24]
# CEA=[0,1]
# sets=["Train","Test"]
# count=0
# np.random.seed(5)
builds=["B002","B011","B082","B083","B084","B085","B086","B087","B088","B089","B090","B096","B097","B098","B099",
         "B100","B101","B115","B116","B123","B124","B125","B126","B127","B128","B129","B131","B132","B133","B138",
         "B198","B199","B200","B201","B202","B203"]

#rows=np.round(a=np.random.uniform(high=10000,low=1,size=200),decimals=0)
for b in np.arange(len(builds)):
    #print("Hour "+str(Hour[h])+" Layer "+str(Lay[l])+" Nodes "+str(NodnoCEA[n])+" AR "+str(AR[ar]))
    #print("Hour "+str(Hour[h])+" Layer "+str(Lay[l])+" Nodes "+str(NodsiCEA[n])+" AR "+str(AR[ar]))
    file0="day"+str(builds[b])+"sarima"
    file1="week"+str(builds[b])+"sarima"
    dataplot0=pd.read_csv(finalloc+file0+".csv",sep=",")
    dataplot1=pd.read_csv(finalloc+file1+".csv",sep=",")
   
    fig0=plt.figure(figsize=(8,6))
    plt.plot(np.array(dataplot0.iloc[:,1]),label='Forecast')
    plt.plot(np.array(dataplot0.iloc[:,2]),label='Real')
    plt.grid(True)
    title0='Prediction for SARIMA Model for building '+str(builds[b])+' HourForec: 24'
    plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
    plt.ylabel('Energy (scaled)',fontsize=15)
    plt.xlabel('Hours',fontsize=15)
    plt.legend(loc='lower right',prop={'size': 10})
    plt.tight_layout()
    fig0.savefig(finalloc+'/Plots thesis/'+file0+'.png')
    plt.close()
    
    fig1=plt.figure(figsize=(8,6))
    plt.plot(np.array(dataplot1.iloc[:,1]),label='Forecast')
    plt.plot(np.array(dataplot1.iloc[:,2]),label='Real')
    plt.grid(True)
    title0='Prediction for SARIMA Model for building '+str(builds[b])+' HourForec: 168'
    plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
    plt.ylabel('Energy (scaled)',fontsize=15)
    plt.xlabel('Hours',fontsize=15)
    plt.legend(loc='lower right',prop={'size': 10})
    plt.tight_layout()
    fig1.savefig(finalloc+'/Plots thesis/'+file1+'.png')
    plt.close()