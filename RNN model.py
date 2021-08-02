'''import libraries and packages'''

import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing as skp
import datetime 
import pandas as pd 
import tensorflow as tf 
import os 

#Data location
#files_location =   os.getcwd()     #for running in linux server
files_location =  'D:/ThesisExperiments' # for running locally


#Fixed parameters
numberofepochs = 300    #epochs for all models
seeds=[7] 		#for reproducing the model
learning_rates= [3/1000]#helps define training speed
display_step=100	#to show training evolution
number_nodes =[1,0,-1]  #changing the number of nodes
hours_forecast=[1,12,24]#changing the hours to forecast
number_layers=[8,5,2]	#changing the number of layers
simulated_vars=[0,1]	#including or not simulated variables
groupsofbuildings=[9,0,1,2,3,4,5,6,7] #using clustered buildings, 9 for all buildings
typecell=['LSTM','GRU'] #changing the cell type in the architecture


startedall=datetime.datetime.now() #starting time to determine duration


def readfiles(FileLocation=files_location, group=9):
'''
This function defines the route for each set of data and according to the 
clustered group the list of routes change

Arguments:
FileLocation 	--the route to the folder where all data is located
group 		--the group to train the model, according to cluster model 
		default=9 for all buildings without clustering

Returns: 	2 tuples
files 		--with the names of each data file for each building
location 	--with the location of each set of data

'''


    locationweather= FileLocation + '/OriginalData/Weather/'
    locationgeometry = FileLocation + '/OriginalData/Geometry/'
    locationtime = FileLocation + '/OriginalData/Time/'
    locationsimulated = FileLocation + '/OriginalData/Simulated/'
    locationenergy = FileLocation + '/OriginalData/Energy/Single Energy/'
    
    filesweather=os.listdir(locationweather)
    filesweather=filesweather[4:44]
    filesgeometry=sorted(os.listdir(locationgeometry))
    filesgeometry=filesgeometry[0:40]
    filestime=os.listdir(locationtime)
    filesenergy=sorted(os.listdir(locationenergy))
    filesenergy=filesenergy[0:40]
    filessimulated=filesgeometry
    group0=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,34]
    group1=[19,20,21,22,23,25,26,27,28,29,32,33,35,37,38,39]
    group2=[17,18,30,31,36]
    group3=[21,25,26,28,29,31,32,33,37]
    group4=[0,1,19,20,22,23,24,27,34,35,39]
    group5=[17,18,30,36]
    group6=[38]
    group7=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    
    if group==0:
        filesgeometry=[filesgeometry[n] for n in group0]
        filesenergy=[filesenergy[n] for n in group0]
        filessimulated=[filessimulated[n] for n in group0]
    elif group==1:
        filesgeometry=[filesgeometry[n] for n in group1]
        filesenergy=[filesenergy[n] for n in group1]
        filessimulated=[filessimulated[n] for n in group1]
    elif group==2:
        filesgeometry=[filesgeometry[n] for n in group2]
        filesenergy=[filesenergy[n] for n in group2]
        filessimulated=[filessimulated[n] for n in group2]
    elif group==3:
        filesgeometry=[filesgeometry[n] for n in group3]
        filesenergy=[filesenergy[n] for n in group3]
        filessimulated=[filessimulated[n] for n in group3]
    elif group==4:
        filesgeometry=[filesgeometry[n] for n in group4]
        filesenergy=[filesenergy[n] for n in group4]
        filessimulated=[filessimulated[n] for n in group4]
    elif group==5:
        filesgeometry=[filesgeometry[n] for n in group5]
        filesenergy=[filesenergy[n] for n in group5]
        filessimulated=[filessimulated[n] for n in group5]
    elif group==6:
        filesgeometry=[filesgeometry[n] for n in group6]
        filesenergy=[filesenergy[n] for n in group6]
        filessimulated=[filessimulated[n] for n in group6]
    elif group==7:
        filesgeometry=[filesgeometry[n] for n in group7]
        filesenergy=[filesenergy[n] for n in group7]
        filessimulated=[filessimulated[n] for n in group7]
    elif group==9:
        filesweather=filesweather
        filesgeometry=filesgeometry
        filestime=filestime
        filesenergy=filesenergy
        filessimulated=filessimulated
    
    files=filesweather,filesgeometry,filestime,filesenergy,filessimulated
    locations=locationweather,locationgeometry,locationtime,locationsimulated,locationenergy
    return files, locations



def CreateDataFilesRNN(FileLocation=files_location,group=9,time_steps_output=24, weathervars=[0,2,9],
		       timevars=[0,1,5],geometryvars=[15,16,17,18,19,20,21],
		       simulatedvars=[2,3,7,54,55,79,84,97],shuffle=False,scaler=False,
		       time_steps_input=24,forecast=1):
'''
This function creates the training and testing sets according to the selected variables to include in the model

Arguments:
FileLocation 	-- the route to the folder where all data is located
group 		-- the group to train the model, according to cluster model. Default=9 for all buildings 
time_steps_output -- number of hours to output from model. Default=24 
weathervars 	--index for weather variables to include in the model. Default=[0,2,9]
timevars 	--index for time variables to include in the model. Default=[0,1,5]
geometryvars 	--index for geometry variables to include in the model. Default=[15,16,17,18,19,20,21]
simulatedvars 	--index for simulated variables to include in the model. Default=[2,3,7,54,55,79,84,97]
shuffle 	--boolean, includes or not shuffling the data 
scaler 		--boolean, includes or not scaling the data 
time_steps_input --number of hours to include as time dimension in the RNN model
forecast	--hours ahead to forecast

Returns:
allinputtrainrnn	--input data for training set
alloutputtrainrnn	--output data for training set
allinputtestrnn		--input data for testing set
alloutputtestrnn	--output data for testing set

'''    
    files, locations=readfiles(FileLocation=FileLocation,group=group)
    locationweather,locationgeometry,locationtime,locationsimulated,locationenergy=locations
    filesweather,filesgeometry,filestime,filesenergy,filessimulated=files

    numvars=len(timevars)+len(weathervars)+len(simulatedvars)+len(geometryvars)
    numregs=(24*365)-forecast
    numregsrnn=numregs-time_steps_output
    
    allinput=np.zeros((0,numvars))
    alloutput=np.zeros((0,1))
    allinputrnn=np.zeros(((numregsrnn*len(filesgeometry)),time_steps_input,numvars))
    alloutputrnn=np.zeros(((numregsrnn*len(filesgeometry)),time_steps_output))
    allinputtrainrnn=allinputtestrnn=np.zeros((0,time_steps_input,numvars))
    alloutputtrainrnn=alloutputtestrnn=np.zeros((0,time_steps_output))
    
    for i in np.arange(len(filesgeometry)):
        builweat=np.asarray((pd.read_csv(locationweather+filesweather[1], sep=",")),dtype=np.float32)
        builgeom=np.asarray((pd.read_csv(locationgeometry+filesgeometry[i], sep=",")),dtype=np.float32)
        builtime=np.asarray((pd.read_csv(locationtime+filestime[1], sep=",")),dtype=np.float32)
        builsimul=np.asarray((pd.read_csv(locationsimulated+filessimulated[i], sep=",")))
        builenergy=np.asarray((pd.read_csv(locationenergy+filesenergy[i],sep=",").Ef_kWh),dtype=np.float32)
        
        weatvar=builweat[:-forecast,weathervars]
        geomvar=builgeom[:-forecast,geometryvars]
        timevar=builtime[:-forecast,timevars]
        simulvar=builsimul[:-forecast,simulatedvars]
        energyvar=builenergy[forecast:]
        #print(energyvar.shape)
        
        invars=np.column_stack((np.column_stack((np.column_stack((timevar,weatvar)),simulvar)),geomvar))
        
        allinput=np.append(allinput,invars,axis=0)
        alloutput=np.append(alloutput,energyvar.reshape(-1,1),axis=0)
        #print(allinput.shape)
        #print(alloutput.shape)
        
        if i==(len(filesgeometry)-1):
            if scaler==True:
                scalerinput=skp.MinMaxScaler((0.0001,1))
                scalerinput.fit(allinput)
                allinput=scalerinput.transform(allinput)
                scaleroutput=skp.MinMaxScaler((0.0001,1))
                scaleroutput.fit(alloutput)
                alloutput=scaleroutput.transform(alloutput)
            
    for j in range(np.shape(allinputrnn)[0]): 
        alloutputrnn[j,:]=alloutput[j:(j+time_steps_output),0]
        for k in range(np.shape(allinput)[1]):
            allinputrnn[j,:,k]=allinput[j:(j+time_steps_input),k]
                
    for l in np.arange(len(filesgeometry)):  
        intr,inte,outtr,outte=train_test_split(allinputrnn[(l*numregs):(l*numregs+numregs),:,:],
                                               alloutputrnn[(l*numregs):(l*numregs+numregs),:],
                                               test_size=0.3,shuffle=shuffle,random_state=seeds)
        allinputtrainrnn=np.append(allinputtrainrnn,intr,axis=0)
        allinputtestrnn=np.append(allinputtestrnn,inte,axis=0)
        alloutputtrainrnn=np.append(alloutputtrainrnn,outtr,axis=0)
        alloutputtestrnn=np.append(alloutputtestrnn,outte,axis=0)
            
    return allinputtrainrnn,alloutputtrainrnn,allinputtestrnn,alloutputtestrnn



def random_mini_batches(X, Y, mini_batch_size = 256, seed = seeds):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            
    m = X.shape[0]             
    batchx = []
    batchy = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    
    num_complete_minibatches = np.floor(m/mini_batch_size)

    for k in np.arange(num_complete_minibatches):
        #mini_batch_X = shuffled_X[int((k*mini_batch_size)):int((k*mini_batch_size+mini_batch_size)),:]
        mini_batch_X = shuffled_X[int((k*mini_batch_size)):int((k*mini_batch_size+mini_batch_size)),:]
        mini_batch_Y = shuffled_Y[int((k*mini_batch_size)):int((k*mini_batch_size+mini_batch_size)),:]
        batchx.append(mini_batch_X) 
        batchy.append(mini_batch_Y)
    
    if m % mini_batch_size != 0:
        mini_batch_X = X[int(num_complete_minibatches * mini_batch_size) : m,:]
        mini_batch_Y = Y[int(num_complete_minibatches * mini_batch_size) : m,:]
        batchx.append(mini_batch_X) 
        batchy.append(mini_batch_Y)
    
    return batchx, batchy 


def trainingnetworkRNN(cellrnntype='LSTM',num_layers=2,nodesvariation=0,numberofepochs=numberofepochs,seed=seeds,
                       batchsize=256,learning_rate=learning_rates,intrain=0,outtrain=0,intest=0,outtest=0):
'''
Use the defined architecture to build the network and then train the model using the parameters given

Arguments:
cellrnntype 	--the kind of cell for the recurrent architecture
num_layers	--the number of layers in the network architecture
nodesvariation	--the number of nodes per layer to include in the architecture
numberofepochs	--the number of epochs for training the model
seed		--the seed for pseudo-random values for initializing the weights of the nodes
batchsize 	--the number of rows to include in the minibathc training
learning_rate	--the learning rate for the model training 
intrain		--the input set for training the model
outtrain	--the output set for training the model
intest		--the input set for testing the model
outtest		--the output set for testing the model

Returns:
returndict	--a dictionary containing the loss evolution in training and testin sets, also the error values in both cases
predictionstrain--the predictions for training set
predictionstest	--the predictions for testing set

'''
    regs,time_steps_input,dim_inputs=intest.shape
    time_steps_output=outtest.shape[1]
    
    if num_layers==2:
        num_nodes=[dim_inputs,time_steps_output]
    elif num_layers==5:
        num_nodes=[dim_inputs,dim_inputs,dim_inputs,dim_inputs,time_steps_output]
    elif num_layers==8:
        num_nodes=[dim_inputs,dim_inputs,dim_inputs,dim_inputs,dim_inputs,dim_inputs,dim_inputs,time_steps_output]
    
    if nodesvariation==-1:
        num_nodes=np.add(num_nodes,-10)
    elif nodesvariation==1:
        num_nodes=np.add(num_nodes,10)
        
    num_nodes[0]=dim_inputs
    num_nodes[-1]=time_steps_output
    
    X=tf.placeholder(tf.float32,[None,time_steps_input,dim_inputs]) 
    Y=tf.placeholder(tf.float32,[None,time_steps_output])

    def rnn_cells(celltype='LSTM',nodes=14): 
        if celltype=='LSTM':
            return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(nodes)#,state_is_tuple=True)
            #return tf.contrib.cudnn_rnn.
            #return tf.nn.rnn_cell.LSTMCell(nodes,state_is_tuple=True)
        elif celltype=='GRU':
            #return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(nodes)
            return tf.nn.rnn_cell.GRUCell(nodes)

    cells=tf.nn.rnn_cell.MultiRNNCell([rnn_cells(celltype=cellrnntype,nodes=nn) for nn in num_nodes])
    outputs, states = tf.nn.dynamic_rnn(cells,X,dtype=tf.float32)
    if cellrnntype=='LSTM':
        preds=states[-1][1]
    elif celltype=='GRU':
        preds=states[-1][0]

    loss=tf.reduce_sum(tf.losses.mean_squared_error(labels=Y,predictions=preds))*100000
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)
    losstrainfile=[]
    mapetrain=[]
    losstestfile=[]
    mapetest=[]
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.set_random_seed(seed)
        for step in range(numberofepochs):
            numbatches=int(np.ceil(np.shape(intrain)[0]/batchsize))
            minibatch_inp, minibatch_outp=random_mini_batches(X=intrain,Y=outtrain,mini_batch_size=batchsize)

            for mi in np.arange(numbatches):
                minibatch_in=minibatch_inp[mi]
                minibatch_out=minibatch_outp[mi]
                
                sess.run([train], feed_dict={X: minibatch_in, Y: minibatch_out})

                if ((mi%(display_step*100)==0) and (step % display_step == 0)):
                    print('training for minibatch ' + str(mi) +' in epoch '+str(step)+' time '+str(datetime.datetime.now()))
                    mse=loss.eval(feed_dict={X:intrain,Y:outtrain})
                    print(step, "\tMSE",mse)
            
            predictions, losses=sess.run([preds, loss], feed_dict={X:intrain,Y:outtrain})
            losstrainfile.append(losses)
            mapetr=100*np.sum(np.absolute(np.divide(np.subtract(outtrain,predictions),outtrain)))/np.shape(outtrain)[0]
            mapetrain.append(mapetr)
            predictions2, losses2=sess.run([preds, loss],feed_dict={X:intest,Y:outtest})
            losstestfile.append(losses2)
            mapete=100*np.sum(np.absolute(np.divide(np.subtract(outtest,predictions2),outtest)))/np.shape(outtest)[0]
            mapetest.append(mapete)
            
    returndict={'losestrainRNN':losstrainfile,'losestestRNN':losstestfile,'mapetrainRNN':mapetrain,'mapetestRNN':mapetest}
    return returndict,predictions,predictions2




#this part iterates over the whole set of variations in architectures or inputs, also write the results of the models in csv files
#using the name of each file for identify the correspondant model

times=[]
models=[]
for gr in groupsofbuildings:
    for hf in hours_forecast:
        for k in simulated_vars:
            if k==0:
                simvar=[]
            if k==1:
                simvar=[2,3,7,54,55,79,84,97]
            buildata=CreateDataFilesRNN(FileLocation=files_location,scaler=True,shuffle=True,group=gr,time_steps_input=24,
                                        time_steps_output=24,forecast=hf,simulatedvars=simvar)
            inptrain,outptrain,inptest,outptest=buildata 
            outtrain=pd.DataFrame(outptrain)
            outtrain.to_csv(files_location +'/OriginalData/ResultsBprevious/ResultsRNN24/RealTrainOut'+str(hf)+'Hour'+str(gr)+'gr.csv')
            outtest=pd.DataFrame(outptest)
            outtest.to_csv(files_location +'/OriginalData/ResultsBprevious/ResultsRNN24/RealTestOut'+str(hf)+'Hour'+str(gr)+'gr.csv')
            for i in number_nodes: 
                for nl in number_layers:
                    for tc in typecell:
                        times.append(str(datetime.datetime.now()))
                        modelran='RNN'+str(tc)+'TimeStepsInpu24'+'group'+str(gr)+'*hourout'+str(hf)+'*simulvar'+str(k)+'*layers'+str(nl)+'*nodes'+str(i)
                        models.append(modelran)
                        tf.reset_default_graph()
                        mapeloss,predtrain,predtest=trainingnetworkRNN(cellrnntype=tc,num_layers=nl,nodesvariation=i,numberofepochs=numberofepochs,
								       seed=seeds,batchsize=256,intrain=inptrain,outtrain=outptrain,intest=inptest,
								       outtest=outptest)
                        RNNcolnamestest=['RNNpredtest'+str(ct) for ct in np.arange(24)]
                        RNNcolnamestrain=['RNNpredtrain'+str(ct) for ct in np.arange(24)]
                        RNNlosesmapeseed=pd.DataFrame(mapeloss)
                        RNNpredicttest=pd.DataFrame(predtest,columns=RNNcolnamestest)
                        RNNpredicttrain=pd.DataFrame(predtrain,columns=RNNcolnamestrain)
                        del mapeloss,predtrain,predtest
                        
                        RNNlosesmapeseed.to_csv(files_location +'/OriginalData/ResultsBprevious/ResultsRNN24/GR9/LosesLay'+str(nl)+'Hour'+str(hf)+'Nod'+str(i)+'Simvar'+str(k)+'Cell'+str(tc)+'gr'+str(gr)+'.csv')
                        RNNpredicttrain.to_csv(files_location +'/OriginalData/ResultsBprevious/ResultsRNN24/GR9/PredTrainLay'+str(nl)+'Hour'+str(hf)+'Nod'+str(i)+'Simvar'+str(k)+'Cell'+str(tc)+'gr'+str(gr)+'.csv')
                        RNNpredicttest.to_csv(files_location +'/OriginalData/ResultsBprevious/ResultsRNN24/GR9/PredTestLay'+str(nl)+'Hour'+str(hf)+'Nod'+str(i)+'Simvar'+str(k)+'Cell'+str(tc)+'gr'+str(gr)+'.csv')

duration=pd.DataFrame({'models':models,'times':times})
duration.to_csv(files_location +'/OriginalData/ResultsRNN24/timesRNNmodels.csv')

finishedall=datetime.datetime.now() #create a variable with the starting time
print('All started at: '+str(startedall)+' and all finished at: '+str(finishedall))
