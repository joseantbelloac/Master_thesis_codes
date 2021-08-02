
import numpy as np #for working with matrices and vectors easily
from sklearn.model_selection import train_test_split #for shuffling and splitting the data into test and train sets
from sklearn import preprocessing as skp
import datetime #to see the begining and ending time of the code
import pandas as pd #for reading quickly the data files
import tensorflow as tf #for implementing the neural networks
import os #to get the current working directory

#Data location
#files_location =   os.getcwd()     #for running in linux server
files_location =  'D:/ThesisExperiments' # for running locally

#Fixed parameters
numberofepochs = 300    #epochs for all models
seeds=[7] 		#for reproducing the model
learning_rates= [3/1000]#helps define training speed
display_step=100	#to show training evolution
numnodes=[31,21,11]	#changing the number of nodes
hourout=[24,12,1]	#changing the hours to forecast
ARenergy=[24,12,0]	#optional inclusion of autoregressive terms
optimizers=[2]		#optional change of optimizer
numlayers=[8,5,2]	#changing the number of layers
dropout=[1.,1.,1.,1.,1.,1.,1.,1.,1.] #posibility of including dropout proportion for each layer
simulvar=[0,1] 		#including or not simulated variables
groupsofbuildings=[9,0,1,2,3,4,5,6,7] #using clustered buildings, 9 for all buildings

startedall=datetime.datetime.now() #create a variable with the starting time


def CreateDataFiles(FileLocation = fileslocation,weathervar=[0,2,9],geometryvar=[15,16,17,18,19,20,21],
                    timevar=[0,1,5],simulatedvar=[2,3,7,54,55,79,84,97],energyvarAR=[],shuffle=False,
                    energyvarOUT=np.arange(24), scaler=False,group=9):
    '''
    This function read, sort, shuffle and scale the input and output data according to the input variables 
    indicated and the output length wanted
    
    Arguments: 
    FileLocation --location of the data files 
    weathervar   --index of weather variables to include in the model  
    geometryvar  --index of geometry variables to include in the model  
    timevar      --index of time variables to include in the model  
    simulatedvar --index of simulated variables to include in the model  
    energyvarAR  --index of autoregressive variables to include in the model  
    energyvarOUT --index of output variable length in the model  
    shuffle      --boolean indicating if the data will be shuffled or not
    scaler       --boolean indicating if the data will be scaled or not 
    group        --the group to train the model, according to cluster model, default=9 for all buildings
    
    Returns:
    Dictionary containing input and output data sorted for the model wanted
    
    '''
    locationweather= FileLocation + '/OriginalData/Weather/'
    locationgeometry = FileLocation + '/OriginalData/Geometry/'
    locationtime = FileLocation + '/OriginalData/Time/'
    locationsimulated = FileLocation + '/OriginalData/Simulated/'
    locationenergy = FileLocation + '/OriginalData/Energy/AR24/'
    
    
    filesweather=os.listdir(locationweather)
    filesweather=filesweather[4:44]
    filesgeometry=sorted(os.listdir(locationgeometry))
    filesgeometry=filesgeometry[0:40]
    filestime=os.listdir(locationtime)
    filesenergy=sorted(os.listdir(locationenergy))
    filesenergy=filesenergy[0:40]
    filessimulated=filesgeometry
    #filesenergy=os.listdir(locationenergy)
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
        
    numvars=len(timevar)+len(weathervar)+len(geometryvar)+len(simulatedvar)+len(energyvarAR)
    energyvarOUT=np.add(energyvarOUT,len(energyvarAR))
    
    buildinputrain=buildinpuvalid=buildinputest=np.zeros((0,numvars))
    eneroutptrain=eneroutpvalid=eneroutptest=np.zeros((0,len(energyvarOUT)))
    
    for i in np.arange(0,len(filesgeometry)):#np.arange(0,40):
        builweat=pd.read_csv(locationweather+filesweather[1], sep=",")
        builweat=np.asarray((builweat))
        builgeom=pd.read_csv(locationgeometry+filesgeometry[i], sep=",")
        builgeom=np.asarray((builgeom))
        builtime=pd.read_csv(locationtime+filestime[1], sep=",")
        builtime=np.asarray((builtime))
        builsimulated=pd.read_csv(locationsimulated+filessimulated[i], sep=",")
        builsimulated=np.asarray((builsimulated))
        
        builenergy=pd.read_csv(locationenergy+filesenergy[i],sep=",")
        builenergy=np.asarray((builenergy))
        
        datastack=np.column_stack((builtime[(0+len(energyvarAR)):(8760-48-1+len(energyvarAR)),timevar],
                                   builweat[(0+len(energyvarAR)):(8760-48-1+len(energyvarAR)),weathervar],
                                   builgeom[(0+len(energyvarAR)):(8760-48-1+len(energyvarAR)),geometryvar],
                                   builsimulated[(0+len(energyvarAR)):(8760-48-1+len(energyvarAR)),simulatedvar],
                                   builenergy[1:,energyvarAR]))
        
        inputrain, inputest, outptrain, outptest = train_test_split(datastack,builenergy[1:,energyvarOUT],
                                                                    train_size=0.7,test_size=0.3, random_state=8,
                                                                    shuffle=shuffle)
        inpuvalid=inputrain[0:500,:]
        outpvalid=outptrain[0:500,:]
        inputrain=inputrain[500:,:]
        outptrain=outptrain[500:,:]
        
        buildinputrain=np.append(buildinputrain,inputrain,axis=0)
        buildinpuvalid=np.append(buildinpuvalid,inpuvalid,axis=0)
        buildinputest=np.append(buildinputest,inputest,axis=0)
        eneroutptrain=np.append(eneroutptrain,outptrain,axis=0)
        eneroutpvalid=np.append(eneroutpvalid,outpvalid,axis=0)
        eneroutptest=np.append(eneroutptest,outptest,axis=0)
       
    originalinput=np.append(buildinputrain,buildinpuvalid, axis=0)
    originalinput=np.append(originalinput,buildinputest,axis=0)
    originaloutput=np.append(eneroutptrain,eneroutpvalid,axis=0)
    originaloutput=np.append(originaloutput,eneroutpvalid,axis=0)
    if scaler==True:
        allinput=np.append(buildinputrain,buildinpuvalid, axis=0)
        allinput=np.append(allinput,buildinputest,axis=0)
        scalerinput=skp.MinMaxScaler((0.0001,1))
        scalerinput.fit(allinput)
        alltransinput=scalerinput.transform(allinput)
        buildinputrain=alltransinput[0:np.shape(buildinputrain)[0],:]
        buildinpuvalid=alltransinput[np.shape(buildinputrain)[0]:(np.shape(buildinputrain)[0]+np.shape(buildinpuvalid)[0]),:]
        buildinputest=alltransinput[(np.shape(buildinputrain)[0]+np.shape(buildinpuvalid)[0]):(np.shape(buildinputrain)[0]+np.shape(buildinpuvalid)[0]+np.shape(buildinputest)[0]),:]
        alloutput=np.append(eneroutptrain,eneroutpvalid,axis=0)
        alloutput=np.append(alloutput,eneroutptest,axis=0)
        scaleroutput=skp.MinMaxScaler((0.0001,1))
        scaleroutput.fit(alloutput)
        alltransoutput=scaleroutput.transform(alloutput)
        eneroutptrain=alltransoutput[0:np.shape(eneroutptrain)[0],:]
        eneroutpvalid=alltransoutput[np.shape(eneroutptrain)[0]:(np.shape(eneroutptrain)[0]+np.shape(eneroutpvalid)[0]),:]
        eneroutptest=alltransoutput[(np.shape(eneroutptrain)[0]+np.shape(eneroutpvalid)[0]):(np.shape(eneroutptrain)[0]+np.shape(eneroutpvalid)[0]+np.shape(eneroutptest)[0]),:]
    
    return {'inptrain':buildinputrain,'inpval':buildinpuvalid,'inptest':buildinputest,'outptrain':eneroutptrain,
            'outpval':eneroutpvalid,'outptest':eneroutptest}    

def NNParametersCreation(entry=1,exit=1,dl1=0,dl2=0,dl3=0,dl4=0,dl5=0,dl6=0,dl7=0,dl8=0, wseed=seeds):
    '''
    This function creates the placeholders and variables for neural network architecture, it includes the inputs 
    and outputs, the weights of hidden layers and the biases, the scaler and offset for normalization and an option
    for dropout in each layer for regularization
    
    Arguments:
    entry -- corresponds to the amount of variables to include as input in the forecasting model
    exit  -- corresponds to the output length
    dl#   -- corresponds to the number of nodes for each one of the maximum 8 layers considered
    wseeds-- the seed value for initializing random variables
    
    Returns:
    weightsnn --dictionary containing the weights of each layer 
    biasesnn  --dictionary containing the bisases of each layer 
    inputsnn  --placeholder with dimension of number of input variables
    outputsnn --placeholder with dimension of output's length  
    scalersnn --dictionary containing the scalers for normalization
    offsetsnn --dictionary containing the offsets for normalization 
    drpt#nn   --placeholders for dropouts proportion in each layer
    
    '''
    tf.set_random_seed(wseed) #setting the random seed for the model
    inputsnn=tf.placeholder("float",[None,entry]) #creating a space for the data to use as input set
    outputsnn=tf.placeholder("float",[None,exit]) #creating a space for the data to use as output set
    
    drpt1nn=tf.placeholder("float",name="drpt1")
    drpt2nn=tf.placeholder("float",name="drpt2")
    drpt3nn=tf.placeholder("float",name="drpt3")
    drpt4nn=tf.placeholder("float",name="drpt4")
    drpt5nn=tf.placeholder("float",name="drpt5")
    drpt6nn=tf.placeholder("float",name="drpt6")
    drpt7nn=tf.placeholder("float",name="drpt7")
    drpt8nn=tf.placeholder("float",name="drpt8")
    
    #creating dictionaries with the weights and biases to use for the architecture
    weightsnn = {
    'h1': tf.Variable(tf.divide(tf.random_normal([entry, dl1],seed=(wseed+1)),100),name="h1"),
    'h2': tf.Variable(tf.divide(tf.random_normal([dl1, dl2],seed=(wseed+2)),100),name="h2"),
    'h3': tf.Variable(tf.divide(tf.random_normal([dl2, dl3],seed=(wseed+3)),100),name="h3"),
    'h4': tf.Variable(tf.divide(tf.random_normal([dl3, dl4],seed=(wseed+4)),100),name="h4"),
    'h5': tf.Variable(tf.divide(tf.random_normal([dl4, dl5],seed=(wseed+5)),100),name="h5"),
    'h6': tf.Variable(tf.divide(tf.random_normal([dl5, dl6],seed=(wseed+6)),100),name="h6"),
    'h7': tf.Variable(tf.divide(tf.random_normal([dl6, dl7],seed=(wseed+7)),100),name="h7"),
    'h8': tf.Variable(tf.divide(tf.random_normal([dl7, dl8],seed=(wseed+8)),100),name="h8"),
    'hout': tf.Variable(tf.divide(tf.random_normal([dl8, exit],seed=(wseed+9)),100),name="hout")
    }
    biasesnn = {
    'b1': tf.Variable(tf.zeros([dl1]),name="b1"),
    'b2': tf.Variable(tf.zeros([dl2]),name="b2"),
    'b3': tf.Variable(tf.zeros([dl3]),name="b3"),
    'b4': tf.Variable(tf.zeros([dl4]),name="b4"),
    'b5': tf.Variable(tf.zeros([dl5]),name="b5"),
    'b6': tf.Variable(tf.zeros([dl6]),name="b6"),
    'b7': tf.Variable(tf.zeros([dl7]),name="b7"),
    'b8': tf.Variable(tf.zeros([dl8]),name="b8"),
    'bout': tf.Variable(tf.zeros([exit]),name="bout")
    }
    iweightsnn = {
    'ih1': tf.Variable(tf.divide(tf.random_normal([entry, dl1],seed=(wseed+1)),100),name="ih1"),
    'ih2': tf.Variable(tf.divide(tf.random_normal([dl1, dl2],seed=(wseed+2)),100),name="ih2"),
    'ih3': tf.Variable(tf.divide(tf.random_normal([dl2, dl3],seed=(wseed+3)),100),name="ih3"),
    'ih4': tf.Variable(tf.divide(tf.random_normal([dl3, dl4],seed=(wseed+4)),100),name="ih4"),
    'ih5': tf.Variable(tf.divide(tf.random_normal([dl4, dl5],seed=(wseed+5)),100),name="ih5"),
    'ih6': tf.Variable(tf.divide(tf.random_normal([dl5, dl6],seed=(wseed+6)),100),name="ih6"),
    'ih7': tf.Variable(tf.divide(tf.random_normal([dl6, dl7],seed=(wseed+7)),100),name="ih7"),
    'ih8': tf.Variable(tf.divide(tf.random_normal([dl7, dl8],seed=(wseed+8)),100),name="ih8"),
    'ihout9': tf.Variable(tf.divide(tf.random_normal([dl8, exit],seed=(wseed+9)),100),name="ihout9")
    }
    scalersnn = {
    'g1': tf.Variable(tf.ones([dl1]),name="g1"),
    'g2': tf.Variable(tf.ones([dl2]),name="g2"),
    'g3': tf.Variable(tf.ones([dl3]),name="g3"),
    'g4': tf.Variable(tf.ones([dl4]),name="g4"),
    'g5': tf.Variable(tf.ones([dl5]),name="g5"),
    'g6': tf.Variable(tf.ones([dl6]),name="g6"),
    'g7': tf.Variable(tf.ones([dl7]),name="g7"),
    'g8': tf.Variable(tf.ones([dl8]),name="g8"),
    'gout': tf.Variable(tf.ones([exit]),name="gout")
    }
    offsetsnn = {
    'o1': tf.Variable(tf.zeros([dl1]),name="o1"),
    'o2': tf.Variable(tf.zeros([dl2]),name="o2"),
    'o3': tf.Variable(tf.zeros([dl3]),name="o3"),
    'o4': tf.Variable(tf.zeros([dl4]),name="o4"),
    'o5': tf.Variable(tf.zeros([dl5]),name="o5"),
    'o6': tf.Variable(tf.zeros([dl6]),name="o6"),
    'o7': tf.Variable(tf.zeros([dl7]),name="o7"),
    'o8': tf.Variable(tf.zeros([dl8]),name="o8"),
    'oout': tf.Variable(tf.zeros([exit]),name="oout")
    }

    return weightsnn,biasesnn, inputsnn, outputsnn, scalersnn, offsetsnn, drpt1nn, drpt2nn, drpt3nn, drpt4nn, drpt5nn, drpt6nn, drpt7nn, drpt8nn;



def neural_net(x,weights=None,biases=None,scalers=None, offsets=None,drpt1=1, drpt2=1, drpt3=1, drpt4=1, drpt5=1, 
                drpt6=1, drpt7=1, drpt8=1, layers=2)
    '''
    This function builds the neural network architecture, following the defined parameters and using the previously
    created weights, biases, scalers, offsets and dropout for each layer. It returns the output of the model
    
    Arguments:
    x       --inputs of the model 
    weights --dictionary containing the weights of each layer 
    biases  --dictionary containing the bisases of each layer 
    inputs  --placeholder with dimension of number of input variables
    outputs --placeholder with dimension of output's length  
    scalers --dictionary containing the scalers for normalization
    offsets --dictionary containing the offsets for normalization 
    drpt#   --placeholders for dropouts proportion in each layer
    layers  --indicates the number of layers for building the model
    
    Returns:
    out_layer --the resulting output from the model
    
    '''
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])  
    meanlayer1, varlayer1 = tf.nn.moments(layer_1,[0])
    layer1norm=tf.nn.batch_normalization(layer_1,meanlayer1,varlayer1,offsets['o1'],scalers['g1'],1e-3)
    actlayer_1=tf.nn.relu(layer1norm)    
    remainlayer1=tf.nn.dropout(actlayer_1,drpt1)
    
    layer_2 = tf.add(tf.matmul(remainlayer1, weights['h2']), biases['b2']) 
    meanlayer2, varlayer2 = tf.nn.moments(layer_2,[0])
    layer2norm=tf.nn.batch_normalization(layer_2,meanlayer2,varlayer2,offsets['o2'],scalers['g2'],1e-3)
    actlayer_2=tf.nn.relu(layer2norm)    
    remainlayer2=tf.nn.dropout(actlayer_2,drpt2)
        
    if layers==2:
        out_layer = tf.matmul(remainlayer2, weights['hout']) + biases['bout']
    elif layers==5:
        layer_3 = tf.add(tf.matmul(remainlayer2, weights['h3']), biases['b3']) 
        meanlayer3, varlayer3 = tf.nn.moments(layer_3,[0])
        layer3norm=tf.nn.batch_normalization(layer_3,meanlayer3,varlayer3,offsets['o3'],scalers['g3'],1e-3)
        actlayer_3=tf.nn.relu(layer3norm)    
        remainlayer3=tf.nn.dropout(actlayer_3,drpt3)
        layer_4 = tf.add(tf.matmul(remainlayer3, weights['h4']), biases['b4']) 
        meanlayer4, varlayer4 = tf.nn.moments(layer_4,[0])
        layer4norm= tf.nn.batch_normalization(layer_4,meanlayer4,varlayer4,offsets['o4'],scalers['g4'],1e-3)
        actlayer_4=tf.nn.relu(layer4norm)    
        actlayer_4=tf.nn.leaky_relu(layer4norm)    
        remainlayer4=tf.nn.dropout(actlayer_4,drpt4)
        layer_5 = tf.add(tf.matmul(remainlayer4, weights['h5']), biases['b5']) 
        meanlayer5, varlayer5 = tf.nn.moments(layer_5,[0])
        layer5norm= tf.nn.batch_normalization(layer_5,meanlayer5,varlayer5,offsets['o5'],scalers['g5'],1e-3)
        actlayer_5=tf.nn.relu(layer5norm)    
        remainlayer5=tf.nn.dropout(actlayer_5,drpt5)
        out_layer = tf.matmul(remainlayer5, weights['hout']) + biases['bout'] 
    elif layers==8:
        layer_6 = tf.add(tf.matmul(remainlayer5, weights['h6']), biases['b6']) 
        meanlayer6, varlayer6 = tf.nn.moments(layer_6,[0])
        layer6norm= tf.nn.batch_normalization(layer_6,meanlayer6,varlayer6,offsets['o6'],scalers['g6'],1e-3)
        actlayer_6=tf.nn.relu(layer6norm)    
        remainlayer6=tf.nn.dropout(actlayer_6,drpt6)
        layer_7 = tf.add(tf.matmul(remainlayer6, weights['h7']), biases['b7']) 
        meanlayer7, varlayer7 = tf.nn.moments(layer_7,[0])
        layer7norm= tf.nn.batch_normalization(layer_7,meanlayer7,varlayer7,offsets['o7'],scalers['g7'],1e-3)
        actlayer_7=tf.nn.relu(layer7norm)    
        remainlayer7=tf.nn.dropout(actlayer_7,drpt7)
        layer_8 = tf.add(tf.matmul(remainlayer7, weights['h8']), biases['b8']) 
        meanlayer8, varlayer8 = tf.nn.moments(layer_8,[0])
        layer8norm= tf.nn.batch_normalization(layer_8,meanlayer8,varlayer8,offsets['o8'],scalers['g8'],1e-3)
        actlayer_8=tf.nn.relu(layer8norm)    
        remainlayer8=tf.nn.dropout(actlayer_8,drpt8)
        out_layer = tf.matmul(remainlayer8, weights['hout']) + biases['bout'] #define the last layer with one single node
    return out_layer   


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

def TrainingNetwork(numberofepochs = numberofepochs,inptrain=0,outptrain=0,inputs=0,outputs=0,learning_rate=0,inpval=0, 
                    outpval=0, optimi=0, inputest=0, outputest=0, layers=8, weights=None, biases=None, scalers=None, 
                    offsets=None, inptest=None, outptest=None, display_step=None,dpt1=1, dpt2=1, dpt3=1, dpt4=1, dpt5=1, 
                    dpt6=1, dpt7=1, dpt8=1,drpt1=0,drpt2=0,drpt3=0,drpt4=0,drpt5=0,drpt6=0,drpt7=0,drpt8=0, batchsize=256):
    '''
    This function creates the model and train it with the given conditions in optimizer, learning rate, batch size 
    and neural network architecture. It returns the errors and prediction for training and testing sets
    
    Arguments:
    numberofepochs   --the number of epochs for training the model
    inptrain         --the input set for training the model
    outptrain        --the output set for training the model
    inputs           --the inputs for the model
    outputs          --the outputs for the model
    learning_rate    --the learning rate for the optimizer
    optimi           --select between optimizers considered 
    intest           --the input set for testing the model
    outtest          --the output set for testing the model
    weights          --dictionary containing the weights of each layer 
    biases           --dictionary containing the bisases of each layer 
    inputs           --placeholder with dimension of number of input variables
    outputs          --placeholder with dimension of output's length  
    scalers          --dictionary containing the scalers for normalization
    offsets          --dictionary containing the offsets for normalization 
    drpt#            --placeholders for dropouts proportion in each layer
    layers           --indicates the number of layers for building the model
    batchsize        --the size of each batch for training
    
    Returns: 
    mapeloss         --dictionary containing the loss and error evolution
    predictions      --the forecast for training set
    predictions2     --ther forecast for test set   
    '''   
    enerpred=neural_net(inputs,weights,biases,scalers,offsets,drpt1, drpt2, drpt3, drpt4, drpt5, drpt6, drpt7, drpt8,layers=layers) 
    
    loss=tf.reduce_sum(tf.losses.mean_squared_error(outputs,enerpred))  #calculate the MSE as loss function to optimize
    #regularizer= tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['h4']) + tf.nn.l2_loss(weights['h5']) + tf.nn.l2_loss(weights['h6']) + tf.nn.l2_loss(weights['h7']) + tf.nn.l2_loss(weights['h8'])
    modelloss=loss*100000 #1000*tf.add(loss,regularizer)
    
    if optimi==1:
        learning_rate=np.divide(learning_rate,100)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimi==2:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)   #define the optimizer to train
    elif optimi==3:
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)    
    
    train_op = optimizer.minimize(modelloss)  
    
    losstrainfile=[]
    mapetrain=[]
    losstestfile=[]
    mapetest=[]

    with tf.Session() as sess:  #begin a tf session
        sess.run(tf.global_variables_initializer()) # Run the initializer
        
        for step in range(1, numberofepochs+1): #run through the number of epochs
            batchsize=batchsize
            numbatches=int(np.ceil(np.shape(inptrain)[0]/batchsize))
            minibatch_inp, minibatch_outp=random_mini_batches(X=inptrain,Y=outptrain,mini_batch_size=batchsize)
                        
            for mi in np.arange(numbatches):
                minibatch_in=minibatch_inp[mi]
                minibatch_out=minibatch_outp[mi]
                sess.run([train_op], feed_dict={inputs: minibatch_in, outputs: minibatch_out,drpt1:dpt1, drpt2:dpt2, drpt3:dpt3, drpt4:dpt4, drpt5:dpt5, drpt6:dpt6, drpt7:dpt7, drpt8:dpt8}) #
                if (mi==0 or mi%(display_step*4)==0) and (step % display_step == 0 or step == 1):
                    print('training for minibatch ' + str(mi) +' in epoch '+str(step))
              
            predictions, losses=sess.run([enerpred, modelloss], feed_dict={inputs:inptrain,outputs:outptrain,drpt1:dpt1, drpt2:dpt2, drpt3:dpt3, drpt4:dpt4, drpt5:dpt5, drpt6:dpt6, drpt7:dpt7, drpt8:dpt8}) #
            losstrainfile.append(losses)
            mapetr=100*np.sum(np.absolute(np.divide(np.subtract(outptrain,predictions),outptrain)))/np.shape(outptrain)[0]
            mapetrain.append(mapetr)
            predictions2, losses2=sess.run([enerpred, modelloss],feed_dict={inputs:inptest,outputs:outptest,drpt1:1, drpt2:1, drpt3:1, drpt4:1, drpt5:1, drpt6:1, drpt7:1, drpt8:1})
            losstestfile.append(losses2)
            mapete=100*np.sum(np.absolute(np.divide(np.subtract(outptest,predictions2),outptest)))/np.shape(outptest)[0]
            mapetest.append(mapete)
            
            if np.isnan(predictions).any():
               break            
            #predictiontrainfile[:, count] = np.array(predictions).reshape(np.shape(data)[0])
            if step % display_step == 0 or step == 1:
                print("Epoch " + str(step) + ", TrainLoss= " + str(losses) + ", TestLoss= " + str(losses2))
    
        
        print("Optimization Finished! with learning rate = "+str(learning_rate))
        print('values for predictions in test ' +str(predictions2[0:4]))
        print('real values in test '+str(outptest[0:4]))
        mapeloss={'losestrain':losstrainfile,'losestest':losstestfile,'mapetrain':mapetrain,'mapetest':mapetest}#,'predval':predictions3,'losesval':losses3}
    return mapeloss, predictions, predictions2



#this part iterates over the whole set of variations in architectures or inputs, also write the results of the models in csv files
#using the name of each file for identify the correspondant model
times=[]
models=[]
for gr in groupsofbuildings:
    for ar in ARenergy: #outputs
        for j in hourout:
            for k in simulvar:
                if k==0:
                    simvar=[]
                if k==1:
                    simvar=[2,3,7,54,55,79,84,97]
                buildata=CreateDataFiles(FileLocation=fileslocation,weathervar=[],geometryvar=[],simulatedvar=simvar,timevar=[],
                                         energyvarOUT=np.arange(j),shuffle=True,scaler=True,energyvarAR=np.arange(ar),group=gr)
                inptrain=buildata['inptrain']
                inpval=buildata['inpval']
                inptest=buildata['inptest']
                outptrain=buildata['outptrain']
                outpval=buildata['outpval']
                outptest=buildata['outptest']
                outtrain=pd.DataFrame(outptrain)
                outtrain.to_csv(fileslocation +'/OriginalData/ResultsB/Layers8/AR'+str(ar)+'RealTrainOut'+str(j)+'Hour'+str(gr)+'gr.csv')
                outtest=pd.DataFrame(outptest)
                outtest.to_csv(fileslocation +'/OriginalData/ResultsB/Layers8/AR'+str(ar)+'RealTestOut'+str(j)+'Hour'+str(gr)+'gr.csv')
                outval=pd.DataFrame(outpval)
                outval.to_csv(fileslocation +'/OriginalData/ResultsB/Layers8/AR'+str(ar)+'RealValOut'+str(j)+'Hour'+str(gr)+'gr.csv')
                ninput=np.shape(inptrain)[1] #fixing the input size to know the number of variables
                noutput= np.shape(outptrain)[1]
                for nl in numlayers:
                    mainnodes=inptrain.shape[1]
                    numnode=[(mainnodes-10),mainnodes,(mainnodes+10)]
                    for i in numnode: 
                        print('Nodes '+str(i)+' Layers '+str(nl)+' Hourout '+str(j))
                        for l in seeds: #initialization seed
                            times.append(str(datetime.datetime.now()))
                            modelran='group'+str(gr)+'*AR'+str(ar)+'*hourout'+str(j)+'*simulvar'+str(k)+'*layers'+str(nl)+'*nodes'+str(i)
                            models.append(modelran)
                            tf.reset_default_graph()
                            weights,biases,inputs,outputs,scalers,offsets,drpt1,drpt2,drpt3,drpt4,drpt5,drpt6,drpt7,drpt8=NNParametersCreation(entry=ninput, 
                                                                                                                                               exit=noutput, 
                                                                                                                                               dl1=i,dl2=i, 
                                                                                                                                               dl3=i, dl4=i, 
                                                                                                                                               dl5=i,dl6=i, 
                                                                                                                                               dl7=i, dl8=i, 
                                                                                                                                               wseed=l)
                            mapeloss,predtrain,predtest=TrainingNetwork(numberofepochs = numberofepochs, inptrain=inptrain, 
                                                                        outptrain=outptrain, inputs=inputs, outputs=outputs,
                                                                        learning_rate=learnrates, optimi=2, layers=nl, 
                                                                        weights=weights, biases=biases, scalers=scalers, 
                                                                        offsets=offsets, inptest=inptest, outptest=outptest, 
                                                                        display_step=display_step, dpt1=dropout[0], 
                                                                        dpt2=dropout[1], dpt3=dropout[2], dpt4=dropout[3], 
                                                                        dpt5=dropout[4], dpt6=dropout[5], dpt7=dropout[6],
                                                                        dpt8=dropout[7], drpt1=drpt1, drpt2=drpt2, 
                                                                        drpt3=drpt3, drpt4=drpt4, drpt5=drpt5, drpt6=drpt6, 
                                                                        drpt7=drpt7, drpt8=drpt8)
                            
                            colnamestest=['predtest'+str(ct) for ct in np.arange(j)]
                            colnamestrain=['predtrain'+str(ct) for ct in np.arange(j)]
                            losesmapeseed=pd.DataFrame(mapeloss)
                            predicttest=pd.DataFrame(predtest,columns=colnamestest)
                            predicttrain=pd.DataFrame(predtrain,columns=colnamestrain)

                            del weights
                            del biases
                            del inputs
                            del outputs
                            del scalers
                            del offsets   
                            del mapeloss,predtrain,predtest
                            
                        losesmapeseed.to_csv(fileslocation +'/OriginalData/ResultsB/Layers'+str(nl)+'/AR'+str(ar)+'Loses'+str(nl)+'Hour'+str(j)+'Nod'+str(i)+'Opti'+str(k)+'gr'+str(gr)+'.csv')
                        predicttrain.to_csv(fileslocation +'/OriginalData/ResultsB/Layers'+str(nl)+'/AR'+str(ar)+'Pred'+str(nl)+'TrainHour'+str(j)+'Nod'+str(i)+'Opti'+str(k)+'gr'+str(gr)+'.csv')
                        predicttest.to_csv(fileslocation +'/OriginalData/ResultsB/Layers'+str(nl)+'/AR'+str(ar)+'Pred'+str(nl)+'TestHour'+str(j)+'Nod'+str(i)+'Opti'+str(k)+'gr'+str(gr)+'.csv')

duration=pd.DataFrame({'models':models,'times':times})
duration.to_csv(fileslocation +'/OriginalData/ResultsB/node14timesgrbatch.csv')

finishedall=datetime.datetime.now() #create a variable with the starting time
print('All started at: '+str(startedall)+' and all finished at: '+str(finishedall))





