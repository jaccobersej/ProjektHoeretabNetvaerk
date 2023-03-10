import tensorflow as tf
import keras 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

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

#VI SAMLER HER ALLE DATAMATRICERNE VI SKAL BRUGE TIL AT TRÆNE PÅ I ET BATCH
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

#VI SAMLER HER ALLE DATAMATRICERNE VI SKAL BRUGE TIL AT VALIDERE VORES TRÆNING PÅ I ET BATCH
TestDataMatrices = np.array(dataMatrices)

###DEFINER, HVILKE LAG MODELLEN INDEHOLDER###
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(32, activation='relu'),
                                    tf.keras.layers.Dense(2, activation='softmax')])

###DEFINER HVILKEN OPTIMIZER OG LEARNING RATE DER SKAL BRUGES###
opt = tf.keras.optimizers.Adam(learning_rate=0.000002)

###KLARGØR MODELLEN FØR BRUG###
model.build(input_shape=(None, 6, 8))

###PRINT ET RESUME AF MODELLENS LAG###
model.summary()

#COMPILE MODELLEN SÅ DEN KAN BRUGES###
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(name='cat_acc')])

###TRÆN MODELLEN OG RETURNER EN HISTORIK AF DENS EVALUERINGER###
model_history = model.fit(TrainDataMatrices, np.asarray(oneHotTrainingLabels), epochs=5000, validation_data=(TestDataMatrices, np.asarray(oneHotTestLabels)))

###PLOT HISTORIKKEN AF MODELLENS EVALUERINGER###
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

###GEM MODELLEN I FOLDEREN MODELS/ (RELATIV PATH)###
model.save('models/')

###FORUDSIG AL TEST DATAEN SÅ VI KAN SE HVORDAN DEN KLARER SIG###
test = model.predict(TestDataMatrices)
print(test)

###PLOT VED HJÆLP AF PLOT FUNKTIONEN FRA FØR###
Train_Val_Plot(model_history.history['cat_acc'], model_history.history['val_cat_acc'], model_history.history['loss'], model_history.history['val_loss'])
