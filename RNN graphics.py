'''RNN'''
import numpy as np
import textwrap as tw
import pandas as pd #for reading quickly the data files
import os #to get the current working directory
from sklearn import preprocessing as skp
import matplotlib #for plotting the results of the model
#matplotlib.use('Agg')  #to create image files without showing them
import matplotlib.pyplot as plt #to add values and organize the image
#%matplotlib inline


finalloc='D:/ThesisExperiments/OriginalData/ResultsB previous/ResultsRNN24/'

'''train and test'''

Hour=[24,12,1]
Lay=[2,5,8]
NodnoCEA=[3,13,23]
NodsiCEA=[11,21,31]
Nods=[-1,0,1]
#AR=[24,12,0]
CEA=[0,1]
sets=["Train","Test"]
count=0
np.random.seed(5)
rows=np.round(a=np.random.uniform(high=10000,low=1,size=200),decimals=0)

for h in np.arange(len(Hour)):
    for l in np.arange(len(Lay)):
        for n in np.arange(len(Nods)):
            #for ar in np.arange(len(AR)):
                count+=1
                #print("Hour "+str(Hour[h])+" Layer "+str(Lay[l])+" Nodes "+str(NodnoCEA[n])+" AR "+str(AR[ar]))
                #print("Hour "+str(Hour[h])+" Layer "+str(Lay[l])+" Nodes "+str(NodsiCEA[n])+" AR "+str(AR[ar]))
                file0="Pred"+sets[0]+"Lay"+str(Lay[l])+"Hour"+str(Hour[h])+"Nod"+str(Nods[n])+"Simvar"+str(CEA[0])+"CellLSTMgr9"
                file1="Pred"+sets[0]+"Lay"+str(Lay[l])+"Hour"+str(Hour[h])+"Nod"+str(Nods[n])+"Simvar"+str(CEA[1])+"CellLSTMgr9"
                file2="Pred"+sets[1]+"Lay"+str(Lay[l])+"Hour"+str(Hour[h])+"Nod"+str(Nods[n])+"Simvar"+str(CEA[0])+"CellLSTMgr9"
                file3="Pred"+sets[1]+"Lay"+str(Lay[l])+"Hour"+str(Hour[h])+"Nod"+str(Nods[n])+"Simvar"+str(CEA[1])+"CellLSTMgr9"
                #file1="AR"+str(AR[ar])+"Pred"+str(Lay[l])+sets[0]+"Hour"+str(Hour[h])+"Nod"+str(NodsiCEA[n]+AR[ar])+"Opti"+str(CEA[1])+"gr9"
                #file2="AR"+str(AR[ar])+"Pred"+str(Lay[l])+sets[1]+"Hour"+str(Hour[h])+"Nod"+str(NodnoCEA[n]+AR[ar])+"Opti"+str(CEA[0])+"gr9"
                #file3="AR"+str(AR[ar])+"Pred"+str(Lay[l])+sets[1]+"Hour"+str(Hour[h])+"Nod"+str(NodsiCEA[n]+AR[ar])+"Opti"+str(CEA[1])+"gr9"
                trainfile="Real"+sets[0]+"Out"+str(Hour[h])+"Hour9gr"
                testfile="Real"+sets[1]+"Out"+str(Hour[h])+"Hour9gr"
                #print(file0)
                dataplot0=pd.read_csv(finalloc+"GR9/"+file0+".csv",sep=",")
                dataplot1=pd.read_csv(finalloc+"GR9/"+file1+".csv",sep=",")
                dataplot2=pd.read_csv(finalloc+"GR9/"+file2+".csv",sep=",")
                dataplot3=pd.read_csv(finalloc+"GR9/"+file3+".csv",sep=",")
                traindata=pd.read_csv(finalloc+trainfile+".csv",sep=",")
                testdata=pd.read_csv(finalloc+testfile+".csv",sep=",")
                
                for c in np.arange(10):             
                    fig0=plt.figure(figsize=(8,6))
                    plt.plot(np.array(dataplot0.iloc[int(rows[c]),1:]),label='Forecast')
                    plt.plot(np.array(traindata.iloc[int(rows[c]),1:]),label='Real')
                    plt.grid(True)
                    #plt.title("\n".join(tw.wrap(text='Loss evolution for Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" AR:"+str(AR[ar])+" NO CEA")),fontsize=15)
                    title0='Train performance for LSTM Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" NO CEA"
                    plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
                    plt.ylabel('Energy (scaled)',fontsize=15)
                    plt.xlabel('Hours',fontsize=15)
                    plt.legend(loc='lower right',prop={'size': 10})
                    plt.tight_layout()
                    fig0.savefig(finalloc+'/Plots thesis/'+file0+'reg'+str(c)+'.png')
                    plt.close()

                    fig1=plt.figure(figsize=(8,6))
                    plt.plot(np.array(dataplot1.iloc[int(rows[c]),1:]),label='Forecast')
                    plt.plot(np.array(traindata.iloc[int(rows[c]),1:]),label='Real')
                    plt.grid(True)
                    #plt.title("\n".join(tw.wrap(text='Loss evolution for Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" AR:"+str(AR[ar])+" NO CEA")),fontsize=15)
                    title0='Train performance for LSTM Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" CEA:Yes"
                    plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
                    plt.ylabel('Energy (scaled)',fontsize=15)
                    plt.xlabel('Hours',fontsize=15)
                    plt.legend(loc='lower right',prop={'size': 10})
                    plt.tight_layout()
                    fig1.savefig(finalloc+'/Plots thesis/'+file1+'reg'+str(c)+'.png')
                    plt.close()

                    fig2=plt.figure(figsize=(8,6))
                    plt.plot(np.array(dataplot2.iloc[int(rows[c]),1:]),label='Forecast')
                    plt.plot(np.array(testdata.iloc[int(rows[c]),1:]),label='Real')
                    plt.grid(True)
                    #plt.title("\n".join(tw.wrap(text='Loss evolution for Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" AR:"+str(AR[ar])+" NO CEA")),fontsize=15)
                    title0='Test performance for LSTM Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" NO CEA"
                    plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
                    plt.ylabel('Energy (scaled)',fontsize=15)
                    plt.xlabel('Hours',fontsize=15)
                    plt.legend(loc='lower right',prop={'size': 10})
                    plt.tight_layout()
                    fig2.savefig(finalloc+'/Plots thesis/'+file2+'reg'+str(c)+'.png')
                    plt.close()

                    fig3=plt.figure(figsize=(8,6))
                    plt.plot(np.array(dataplot3.iloc[int(rows[c]),1:]),label='Forecast')
                    plt.plot(np.array(testdata.iloc[int(rows[c]),1:]),label='Real')
                    plt.grid(True)
                    #plt.title("\n".join(tw.wrap(text='Loss evolution for Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" AR:"+str(AR[ar])+" NO CEA")),fontsize=15)
                    title0='Test performance for LSTM Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" CEA:Yes"
                    plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
                    plt.ylabel('Energy (scaled)',fontsize=15)
                    plt.xlabel('Hours',fontsize=15)
                    plt.legend(loc='lower right',prop={'size': 10})
                    plt.tight_layout()
                    fig3.savefig(finalloc+'/Plots thesis/'+file3+'reg'+str(c)+'.png')
                    plt.close()
              
                


#losses


# Hour=[1,12,24]
# Lay=[2,5,8]
# NodnoCEA=[3,13,23]
# NodsiCEA=[11,21,31]
# AR=[24,12,0]
# CEA=[0,1]
count=0
for h in np.arange(len(Hour)):
    for l in np.arange(len(Lay)):
        for n in np.arange(len(NodnoCEA)):
            for ar in np.arange(len(AR)):
                count+=1
                #print("Hour "+str(Hour[h])+" Layer "+str(Lay[l])+" Nodes "+str(NodnoCEA[n])+" AR "+str(AR[ar]))
                #print("Hour "+str(Hour[h])+" Layer "+str(Lay[l])+" Nodes "+str(NodsiCEA[n])+" AR "+str(AR[ar]))
                file0="LosesLay"+str(Lay[l])+"Hour"+str(Hour[h])+"Nod"+str(Nods[n])+"Simvar"+str(CEA[0])+"CellLSTMgr9"
                file1="LosesLay"+str(Lay[l])+"Hour"+str(Hour[h])+"Nod"+str(Nods[n])+"Simvar"+str(CEA[1])+"CellLSTMgr9"
                #print(file0)
                dataplot0=pd.read_csv(finalloc+"Loses/GR9/"+file0+".csv",sep=",")
                #print(file1)
                dataplot1=pd.read_csv(finalloc+"Loses/GR9/"+file1+".csv",sep=",")
                
                             
                fig0=plt.figure(figsize=(8,6))
                plt.plot(dataplot0.iloc[:,1],label='Train')
                plt.plot(dataplot0.iloc[:,2],label='Test')
                plt.grid(True)
                #plt.title("\n".join(tw.wrap(text='Loss evolution for Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" AR:"+str(AR[ar])+" NO CEA")),fontsize=15)
                title0='Loss evolution for LSTM Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(Nods[n])+" NO CEA"
                plt.title("\n".join(tw.wrap(title0,45)),fontsize=15)
                plt.ylabel('MSE ',fontsize=15)
                plt.xlabel('Epochs',fontsize=15)
                plt.legend(loc='lower right',prop={'size': 10})
                plt.tight_layout()
                fig0.savefig(finalloc+'/Plots thesis/Loses RNN/'+file0+'.png')
                plt.close()
                
                fig1=plt.figure(figsize=(8,6))
                plt.plot(dataplot1.iloc[:,1],label='Train')
                plt.plot(dataplot1.iloc[:,2],label='Test')
                plt.grid(True)
                #plt.title("\n".join(tw.wrap(text='Loss evolution for Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(NodnoCEA[n])+" AR:"+str(AR[ar])+" NO CEA")),fontsize=15)
                title1='Loss evolution for LSTM Model HourForec:'+str(Hour[h])+' Layer:'+str(Lay[l])+" Nodes:"+str(Nods[n])+" CEA:Yes"
                plt.title("\n".join(tw.wrap(title1,45)),fontsize=15)
                plt.ylabel('MSE ',fontsize=15)
                plt.xlabel('Epochs',fontsize=15)
                plt.legend(loc='lower right',prop={'size': 10})
                plt.tight_layout()
                fig1.savefig(finalloc+'/Plots thesis/Loses RNN/'+file1+'.png')
                plt.close()
                