import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda
import pandas as pd
import os



"""WEIGHTS OG BIAS"""
###DEFINER GLOBALE VARIABLE###
weights = None
biases = None
    


"""DATA BEHANDLING"""
### LOAD DATA FRA CSV FIL###
dataframe = pd.read_csv('HoereData.csv')
TrainData = np.array(dataframe)
TestData = np.array(pd.read_csv('HoereTestData.csv'))

### FORMATER DATA TIL BRUGBAR DATA###
count = 0
dataMatrices = []
tempMatrix = []
oneHotTrainingLabels = []
for row in TrainData:
    count += 1
    tempRow = [row[1], row[2], row[4], row[6], row[7], row[8], row[9], row[10]]
    tempMatrix.append(tempRow)
    if count % 6 == 0:
        dataMatrices.append(tempMatrix)
        if row[3] == 1:
            oneHotTrainingLabels.append([0, row[3]])
        if row[3] == 0:
            oneHotTrainingLabels.append([1, row[3]])
        tempMatrix = []

TrainDataMatrices = np.array(dataMatrices)

count = 0
dataMatrices = []
oneHotTestLabels = []
for row in TestData:
    count += 1
    tempRow = [row[1], row[2], row[4], row[6], row[7], row[8], row[9], row[10]]
    tempMatrix.append(tempRow)
    if count % 6 == 0:
        dataMatrices.append(tempMatrix)
        if row[3] == 1:
            oneHotTestLabels.append([0, row[3]])
        elif row[3] == 0:
            oneHotTestLabels.append([1, row[3]])
        tempMatrix = []

TestDataMatrices = np.array(dataMatrices)
    


"""LAYERS"""
class layer:
    def __init__(self):
        pass
    ###INPUT LAYER (FLATTEN)###
    def Input_flatten(data, weights=None, bias=None, aktiverings_funktion=None):
        flat = data.flatten()
        flat = np.reshape(flat, (len(flat),1))
        return flat
        
    ###FULLY CONNECTED LAYER###
    def Dense(dataVektor, weights, bias, aktiverings_funktion):
        ikke_aktiveret = np.dot(weights, dataVektor)+bias
        aktiveringer = globals()[aktiverings_funktion](ikke_aktiveret)
        return ikke_aktiveret, aktiveringer



class loss:
    def __init__(self):
        pass
    
    def Categorical_crossentropy(aktiveringer, TruthLabels):
        return -np.sum(TruthLabels * np.log(aktiveringer + 10**-100))


"""AKTIVERINGS FUNKTIONER"""
def Relu(z):
    return np.maximum(0, z)

def ReluDerivative(z):
    return np.heaviside(z, 0)

def Softmax(z):
    #Shifter z værdier mod 0 for at sikre numerisk stabilitet, sådan at man overflower mindre og derved får færre NaN værdier.
    shiftz = z - np.max(z)
    return (np.exp(shiftz)/np.exp(shiftz).sum())

def dEntropyLossWRTZ(a, y):
    return np.subtract(Softmax(a), y)
    


"""MODEL FUNKTIONALITET"""

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.weights = None
        self.biases = None
        self.neurons = None
        self.epoch = None

    ###COMPILE MODEL FOR AT LAVE NYE VÆGTE OG BIAS###
    def compile(self):
        self.weights, self.biases = self.initWB(self.neurons)
        
        
    ###LAV NYE WEIGHTS OG BIAS###
    def initWB(self, neuroner):
        weights = []
        for i, j in zip(neuroner[:-1], neuroner[1:]):
            weights.append(3*np.random.randn(j, i))
        weights = np.array(weights, dtype=object)    
        
        biases = []
        for i in neuroner[1:]:
            biases.append(np.zeros((i, 1)))
        biases = np.array(biases, dtype=object)
        return weights, biases
    
    def save(self, savepath):
        np.save(savepath + "weights.npy", self.weights)
        np.save(savepath + "bias.npy", self.biases)
    
    ###LOAD WEIGHTS OG BIAS###
    def load(self, input):
        try:
            print("Loaded weights and bias")
            self.weights = np.load(input + "weights.npy", allow_pickle=True)
            self.biases = np.load(input + "bias.npy", allow_pickle=True)
            """for index, weight in enumerate(self.weights):
                self.weights[index] = weight.T"""
        except:
            print('Ingen gemte weights og/eller bias. Laver nye weights og bias')
            self.weights, self.biases = self.initWB(input)
        
    def predict(self, data):
        if self.layers[0] != layer.Input_flatten:
            tidligereAktiveringer = layer.Input_flatten(data)
        for (anylayer, aktiverings_funktion), weight, bias in zip(self.layers, self.weights, self.biases):
            z, tidligereAktiveringer = anylayer(tidligereAktiveringer, weight, bias, aktiverings_funktion)
        sidsteAktivering = tidligereAktiveringer.flatten()
        return (np.argmax(sidsteAktivering), sidsteAktivering)
    
    def evaluate(self, trainingset, traininglabels, validationset=None, validationlabels=None):
        rigtigeGætTræning = 0
        rigtigeGætValidation = 0
        træningsAcc = 0
        tempTrænLoss = []
        træningsLoss = 0
        validationAcc = 0
        tempValLoss = []
        validationLoss = 0
        
        
        ###UDREGN TRÆNINGS ACCURACY###
        if len(trainingset.shape) > 3:
            for batch in trainingset:
                for image, label in batch:
                    prediction = self.predict(image)
                    gæt = prediction[0]
                    
                    tempTrænLoss.append(loss.Categorical_crossentropy(prediction[1], label))
                    
                    if label[gæt] == 1:
                        rigtigeGætTræning += 1
            træningsAcc = rigtigeGætTræning/(len(trainingset)*len(batch))
            træningsLoss = np.mean(tempTrænLoss, axis=0)
        else:
            for image, label in zip(trainingset, traininglabels):
                prediction = self.predict(image)
                gæt = prediction[0]
                tempTrænLoss.append(loss.Categorical_crossentropy(prediction[1], label))
                
                if label[gæt] == 1:
                    rigtigeGætTræning += 1
            træningsAcc = rigtigeGætTræning/len(trainingset)
            træningsLoss = np.mean(tempTrænLoss, axis=0)
            if validationset[0][0][0] == None:
                print("Epoch: ", self.epoch, "---------" ,"Trænings accuracy: ", træningsAcc, "---------" ,"Trænings Loss: ", træningsLoss)
            
        ###UDREGN VALIDATIONS ACCURACY###    
        if validationset[0][0][0] != None:
            if len(validationset.shape) > 3:
                for batch in validationset:
                    for image, label in zip(batch, validationlabels):
                        prediction = self.predict(image)
                        gæt = prediction[0]
                        
                        tempValLoss.append(loss.Categorical_crossentropy(prediction[1], label))
                        
                        if label[gæt] == 1:
                            rigtigeGætValidation += 1
                validationAcc = rigtigeGætValidation/(len(validationset)*len(batch))
                validationLoss = np.mean(tempValLoss, axis=0)
            else:
                for image, label in zip(validationset, validationlabels):
                    prediction = self.predict(image)
                    gæt = prediction[0]
                    
                    tempValLoss.append(loss.Categorical_crossentropy(prediction[1], label))
                    
                    if label[gæt] == 1:
                        rigtigeGætValidation += 1
                validationAcc = rigtigeGætValidation/len(validationset)
                validationLoss = np.mean(tempValLoss)
                print("Epoch: ", self.epoch, "---------" ,"Trænings accuracy: ", træningsAcc, "---------" ,"Trænings Loss: ", træningsLoss, "---------", "Validations accuracy: ", validationAcc, "---------", "Validations Loss: ", validationLoss)
        
        return træningsAcc, træningsLoss, validationAcc, validationLoss
                
                
    def train(self, trainingSet, trainingLabels, epochs, validationset=None, validationLabels=None, learning_rate=0.1):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        firstMoment_b = [np.zeros(b.shape) for b in self.biases]
        firstMoment_w = [np.zeros(w.shape) for w in self.weights]
        SecondMoment_b = [np.zeros(b.shape) for b in self.biases]
        SecondMoment_w = [np.zeros(w.shape) for w in self.weights]
        history = []
        
        if len(trainingSet.shape) > 3:
            for epoch in range(0, epochs):
                for batch in trainingSet:
                    for image, label in zip(batch, trainingLabels):
                        bias_deltaGrad, weight_deltaGrad = self.backpropagation(image, np.reshape(np.array(label), (2,1)))
                        nabla_b = [nb + dnb/len(batch) for nb, dnb in zip(nabla_b, bias_deltaGrad)] 
                        nabla_w = [nw + dnw/len(batch) for nw, dnw in zip(nabla_w, weight_deltaGrad)] 
                        
                    firstMoment_b = [(beta1 * np.array(firstMoment_b) + (1-beta2) * nb)/(1-beta1) for nb in nabla_b]
                    firstMoment_w = [(beta1 * np.array(firstMoment_w) + (1-beta2) * nw)/(1-beta1) for nw in nabla_w]
                    SecondMoment_b = [(beta2 * np.array(SecondMoment_b) + (1-beta2) * nb**2)/(1-beta2) for nb in nabla_b]
                    SecondMoment_w = [(beta2 * np.array(SecondMoment_w) + (1-beta2) * nw**2)/(1-beta2) for nw in nabla_w]
                    self.weights = np.array([np.subtract(w, learning_rate * (mw/np.sqrt(vw+epsilon))) for w, mw, vw in zip(self.weights, firstMoment_w, SecondMoment_w)], dtype=object)
                    self.biases  = np.array([np.subtract(b, learning_rate * (mb/np.sqrt(vb+epsilon))) for b, mb, vb in zip(self.biases, firstMoment_b, SecondMoment_b)], dtype=object)
                    #print(self.weights[-1])
                    history.append(self.evaluate(trainingSet, validationset))
        else:
            for epoch in range(0, epochs):
                self.epoch = epoch
                for image, label in zip(trainingSet, trainingLabels):
                    bias_deltaGrad, weight_deltaGrad = self.backpropagation(image, np.reshape(np.array(label), (2,1)))
                    nabla_b = [nb + dnb/len(trainingSet) for nb, dnb in zip(nabla_b, bias_deltaGrad)] 
                    nabla_w = [nw + dnw/len(trainingSet) for nw, dnw in zip(nabla_w, weight_deltaGrad)] 
                    
                firstMoment_b = [(beta1 * fmb + (1-beta2) * nb)/(1-beta1) for nb, fmb in zip(nabla_b, firstMoment_b)]
                firstMoment_w = [(beta1 * fmw + (1-beta2) * nw)/(1-beta1) for nw, fmw in zip(nabla_w, firstMoment_w)]
                SecondMoment_b = [(beta2 * smb + (1-beta2) * nb**2)/(1-beta2) for nb, smb in zip(nabla_b, SecondMoment_b)]
                SecondMoment_w = [(beta2 * smw + (1-beta2) * nw**2)/(1-beta2) for nw, smw in zip(nabla_w, SecondMoment_w)]
                #print("ADAM: ", learning_rate * (firstMoment_w[0]/np.sqrt(SecondMoment_w[0]+epsilon)))
                self.weights = [np.subtract(w, learning_rate * (mw/np.sqrt(vw+epsilon))) for w, mw, vw in zip(self.weights, firstMoment_w, SecondMoment_w)]
                self.biases  = [np.subtract(b, learning_rate * (mb/np.sqrt(vb+epsilon))) for b, mb, vb in zip(self.biases, firstMoment_b, SecondMoment_b)]
                #print("WEIGHTS: ", self.weights[0])
                if epoch % 10 == 0:
                    if validationset[0][0][0] != None:
                        history.append(self.evaluate(trainingSet, trainingLabels, validationset, validationLabels))
                    else:
                        history.append(self.evaluate(trainingSet, trainingLabels))
        
        return history
        
    def backpropagation(self, data, label):
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        zList = []
        activationList = []
        
        if self.layers[0] != layer.Input_flatten:
            aktiveringer = layer.Input_flatten(data)
            activationList.append(aktiveringer)
        for (anylayer, aktiverings_funktion), weight, bias in zip(self.layers, self.weights, self.biases):
            #print("aktiveringer: ", aktiveringer.shape, "weight: ", weight.shape, "bias: ", bias.shape)
            z, aktiveringer = anylayer(aktiveringer, weight, bias, aktiverings_funktion)
            zList.append(z)
            activationList.append(aktiveringer)
            
        if self.layers[-1][1] == 'Softmax':
            dLdB = (activationList[-1]-label)
            dLdW = np.dot(dLdB, activationList[-2].T)
            bias_gradient[-1] = dLdB
            weight_gradient[-1] = dLdW
            
        for index in range(0, len(zList)-1):
            dLdB = np.dot(dLdB.T, self.weights[-index-1]).T * globals()[self.layers[-index-2][1]+"Derivative"](zList[-index-2])
            dLdW = np.dot(dLdB, activationList[-index-3].T)
            bias_gradient[-index-2] = dLdB
            weight_gradient[-index-2] = dLdW
            #print("gradient form bias: ", np.array(bias_gradient ,dtype=object)[2].shape)
        return (bias_gradient, weight_gradient)
        

###CONFIG###
inputshape = TrainDataMatrices[0].shape[0] * TrainDataMatrices[0].shape[1]
neuroner = [inputshape, 64, 32, 2]
beta1 = 0.4
beta2 = 0.499
epsilon = 10**-7

test = Model([[layer.Dense, 'Relu'], 
              [layer.Dense, 'Relu'], 
              [layer.Dense, 'Softmax']])


test.neurons = neuroner
test.compile() #kør, hvis netværket ikke har været kørt før og der ikke er nogle vægte eller bias
#test.load("test/")
history = test.train(TrainDataMatrices, oneHotTrainingLabels, 15000, learning_rate=0.01, validationset=TestDataMatrices, validationLabels=oneHotTestLabels)
test.save("test/")
gæt, gæt2 = test.predict(TrainDataMatrices[0])
print(TrainDataMatrices[0][3], gæt, gæt2)
gæt, gæt2 = test.predict(TrainDataMatrices[1])
print(TrainDataMatrices[1][3], gæt, gæt2)

def Train_Val_Plot(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])

    plt.show()
    
history = np.array(history, dtype=object).T
Train_Val_Plot(history[0], history[2], history[1], history[3])

""""NOTER OG TING SOM SKAL HUSKES"""

#loadWB([8, 512,256,128,2]) #input neuroner (inklusiv input og output)
#test.load("test/")