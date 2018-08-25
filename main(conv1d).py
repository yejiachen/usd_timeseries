import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import optimizers, backend
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, Activation
from keras.models import load_model
from keras.layers import RepeatVector, TimeDistributed, LSTM, GRU, Flatten, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.utils.generic_utils import get_custom_objects

from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, adam, Nadam

def swish(x):
    return ((K.sigmoid(x) * x * 1.5))
get_custom_objects().update({'swish': Activation(swish)})



def reshape_data(raw_data, feature, window, forecast_day, training_data_rate, scaler):

    train = pd.read_csv(raw_data)
    train = train[::-1]
    train_data = train[ feature ]

    real_updown = [None]
    for i in range(len(train_data)-1):
        if train_data.iloc[i+1,0]-train_data.iloc[i,0] >= 0:
            real_updown.append(1)
        else:
            real_updown.append(0)

    train_data = scaler.fit_transform(train_data)
    feature_n = train_data.shape[1]

    train_x = np.empty(shape=(0, window, feature_n))
    train_y = np.empty(shape=(0, forecast_day))
    for i in range(len(train_data)-window-forecast_day+1):
        train_x = np.vstack((train_x, train_data[np.newaxis,i:(i+window),:]))
        train_y = np.vstack((train_y, train_data[(i+window):(i+window+forecast_day), 0]))

    train_x, validation_x = train_x[:int(len(train_x)*training_data_rate),:,:], train_x[int(len(train_x)*training_data_rate):,:,:] 
    train_y, validation_y = train_y[:int(len(train_y)*training_data_rate),:], train_y[int(len(train_y)*training_data_rate):,:]
    
    return train_x, validation_x, train_y, validation_y, train, real_updown



def pred_7d(model_saved, validation_x, validation_y, forecast_day, scaler, train, real_updown):
    
    model = load_model(model_saved)
    validation_set_pred = model.predict(validation_x, batch_size=4)

    if scaler == MinMaxScaler():
        validation_set_pred = validation_set_pred * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
        validation_y = validation_y * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
        
    elif scaler == StandardScaler():
        validation_set_pred = validation_set_pred * (scaler.var_[3]**(1/2)) + scaler.mean_[3]
        validation_y = validation_y * (scaler.var_[3]**(1/2)) + scaler.mean_[3]
    
    predictions = np.zeros(shape=(len(validation_y), forecast_day, 5))
    for i in range(len(validation_y)):
        #value of prediction
        predictions[i,:,0] = validation_set_pred[i,:]
        #actual value
        predictions[i,:,1] = validation_y[i,:].reshape(1,forecast_day)
        #prediction of trend
        predictions[i,1:,2] = (predictions[i,1:,0] - predictions[i,:-1,1]) >= 0
        #actual trend
        predictions[i,:,3] = real_updown[int(len(train_x))+window:][i:i+forecast_day]
    
    for i in range(1,len(predictions)):
        predictions[i,0,2] = (predictions[i,0,0] - predictions[i-1,0,1] > 0)
    predictions[0,0,2] = (predictions[0,0,0] - train.iloc[int(len(train_x))+window-1, 1]) > 0
    
    for i in range(len(predictions)):
        #weather prediction of trend is accurate or not
        predictions[i,:,4] = (predictions[i,:,2] == predictions[i,:,3])
        
    return predictions       


def pred_x_stacking(model_saved, validation_x, validation_y, forecast_day, scaler, train, real_updown):
   
    model = load_model(model_saved)
    validation_set_pred = model.predict(validation_x, batch_size=4)

    if scaler == MinMaxScaler():
        validation_set_pred = validation_set_pred * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
        validation_y = validation_y * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
        
    elif scaler == StandardScaler():
        validation_set_pred = validation_set_pred * (scaler.var_[3]**(1/2)) + scaler.mean_[3]
        validation_y = validation_y * (scaler.var_[3]**(1/2)) + scaler.mean_[3]
    
    predictions = np.zeros(shape=(len(validation_y), forecast_day))
    for i in range(len(validation_y)):
        #value of prediction
        predictions[i,:] = validation_set_pred[i,:]
    return predictions

def show_evaluation(predictions,forecast_day):
    for i in range(forecast_day):
        print("Trend accuracy of", i+1,"days after：", predictions[:,i,4].mean())
    print("Model: " + model_saved + " average trend accuracy：", sum([predictions[i,:,4].mean() for i in range(len(predictions))])/len(predictions))


def df_evaluation(predictions,forecast_day):
    eval_dt = np.zeros(shape=(forecast_day+1, 1))
    index = np.zeros(shape=(forecast_day+1, 1))
    
    # evaluate everyday trend accuracy
    for i in range(forecast_day):
        eval_dt[i,0]=predictions[:, i, 4].mean()
    
    # evaluate average trend accuracy
    eval_dt[-1, 0] = sum([predictions[i, :, 4].mean() for i in range(len(predictions))])/len(predictions)
    
    index = ['%s %s'%(n, 'Day') for n in list(range(1, i+2))]
    index.append('Average')

    df = pd.DataFrame(eval_dt, columns = [model_saved])
    df.index = index
    return df

# main
raw_data = "raw_data.csv"
feature = feature
window = 5
forecast_day = 7
training_data_rate = 0.95
scaler = MinMaxScaler()
model_saved = model_saved

# reshape data to time series - 5days predict future 7days
train_x, validation_x, train_y, validation_y, train, real_updown = reshape_data(raw_data, feature, window, forecast_day, training_data_rate, scaler)

# build cnn model
model = Sequential()
model.add(Conv1D(nb_filter=64, filter_length=1, input_shape=(window, feature), activation=swish))
model.add(Conv1D(nb_filter=128, filter_length=1, activation=swish))
model.add(Conv1D(nb_filter=128, filter_length=1, activation=swish))
model.add(Conv1D(nb_filter=64, filter_length=1, activation=swish))
model.add(Dropout(rate=0.6))

model.add(Flatten())
model.add(Dense(49,activation=swish))
model.add(Dense(forecast_day,activation="linear"))

print(model.summary())

# set optimizers & callbacks
adam = optimizers.Adam(lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=1e-6,
    amsgrad=False)

nadam = optimizers.Nadam(lr=0.0001, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=None, 
    schedule_decay=0.001)

model.compile(
    loss = "mean_squared_logarithmic_error",
    optimizer = nadam,
    metrics = ["mean_squared_logarithmic_error"])

checkpoint = ModelCheckpoint(
    filepath = model_saved,
    monitor = "val_loss",
    verbose = 0,
    save_best_only = True,
    mode = "min")

earlystopping = EarlyStopping(
    monitor = "val_loss",
    patience = 200,
    verbose = 1,
    mode = "auto")


 # train deep learning model by using tensorflow backed gpu
start = timeit.default_timer()
train_history = model.fit(
    x=train_x,
    y=train_y,
    epochs=2500,
    validation_data=(validation_x, validation_y),
    batch_size=8,
    shuffle=False,
    verbose=0,
    callbacks = [checkpoint, earlystopping])

# summarize history for loss
plt.figure(figsize=(20,8))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# predict y from validation_x by best-saved model weight
predictions = pred_7d(model_saved,validation_x, validation_y, 7, scaler, train, real_updown)

# evaluation model by trend accuracy
evaluation(predictions,forecast_day)
