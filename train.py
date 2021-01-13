import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization

TRAIN_TEST_SPLIT = 0.25
BATCH_SIZE = 32
EPOCHS = 20
MODEL_FILENAME = "SpeechEmotionRecog-RAVDESS-TESS"

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
#Emotions in the RAVDESS dataset
emotionsRAVDESS = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
emotionsTESS = ['angry', 'disgust', 'fear', 'happy', 'ps', 'sad', 'neutral']
# Emotions to train the network on
trainingEmotions =  list(emotionsRAVDESS.values())
#trainingEmotions = ['neutral', 'sad', 'happy', 'surprised', 'fearful']
#TESS dataset emotions
mapTESStoRAVDESS = {
    'fear':'fearful',
    'ps':'surprised'
}


def extractFeature(fileName, mfcc=True, chroma=True, mel=True, noise=False):
    with soundfile.SoundFile(fileName) as soundFile:
        X = soundFile.read(dtype= "float32")
        sampleRate = soundFile.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y= X, sr=sampleRate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sampleRate).T, axis=0)
            result = np.hstack((result, mel))
        if noise:
            result = noise(result)
    return result

def noise(data, amount=0.005):
    #Adding White Noise.
    noiseAmp = amount*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noiseAmp * np.random.normal(size=data.shape[0])
    return data

# Load RAVDESS data and return features
def loadRAVDESS():
    x, y = [], []
    emotion = 0
    for file in glob.glob(f"{os.getcwd()}\data\\RAVDESS\\Actor_*\*.wav"):
        #for file in glob.glob("{os.path.dirname(os.path.abspath(__file__))}\\data\\Actor_*\\*.wav"):
        fileNameStr = os.path.basename(file)
        fileEmotion = fileNameStr.split("-")
        if(len(fileEmotion) < 3):
            continue
        #skip emotions that are not being observed
        if emotionsRAVDESS[fileEmotion[2]] not in trainingEmotions:
            continue
        emotion = int(fileEmotion[2])
        feature = extractFeature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        #y.append(emotion)
        y.append(emotionsRAVDESS[fileEmotion[2]])
    #return train_test_split(np.array(x), y, test_size= splitTestData, random_state=9)
    x = np.array(x)
    y = np.array(y)
    return x, y

#load TESS dataset and return features
def loadTESS():
    x, y = [], []
    for file in glob.glob(f"{os.path.dirname(os.path.abspath(__file__))}\\data\\TESS\\*.wav"):
        fileNameStr = os.path.splitext(os.path.basename(file))[0]
        fileEmotion = fileNameStr.split("_")[2]    #third str indicates emotion
        fileEmotion = convertTESStoEmotion(fileEmotion)
        if fileEmotion not in trainingEmotions:
            continue        #skip file
        feature = extractFeature(file, mfcc = True, chroma=True, mel=True)
        x.append(feature)
        y.append(fileEmotion)
    #return train_test_split(np.array(x), y, test_size= splitTestData, random_state=9)
    x = np.array(x)
    y = np.array(y)
    return x, y

def loadUserData():
    x, y = [], []
    for file in glob.glob(f"{os.getcwd()}\data\\User\\*.wav"):
        fileNameStr = os.path.basename(file)
        fileEmotion = fileNameStr.split('-')
        if(len(fileEmotion) < 2) and (fileEmotion[1] not in trainingEmotions):
            continue
        feature = extractFeature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(fileEmotion[1])
    x = np.array(x)
    y = np.array(y)
    return x, y
        
def convertTESStoEmotion(emotion):
    if emotion in mapTESStoRAVDESS.keys():
        return mapTESStoRAVDESS[emotion]
    else: 
        return emotion

#example config passed to trainModel.
config = {
    'dataset':'RAVDESS-TESS',
    'test_split':0.25,
    'batch_size':32,
    'epochs':40,
    'activation_function':'tanh',
    'learning_rate':2e-5,
    'decay':1e-6,
    'epsilon':1e-8,
    'layer_1': 300,
    'layer_2': 150,
    'layer_3': 125
    }

def trainModel(model_name, train_config):
    #load dataset
    x, y = [], []
    x = np.array(x)
    y = np.array(y)
    if 'dataset' in train_config:    
        datasets = train_config['dataset'].split('-')
    else:
        return "Error: dataset not found in configuration"
    if 'test_split' in train_config:  
        test_split = train_config['test_split']
    else:
        test_split = 0.25
    if 'batch_size' in train_config:  
        batch_size = train_config['batch_size']
    else:
        batch_size = 64
    if 'epochs' in train_config:  
        epochs = train_config['epochs']
    else:
        epochs = 30
    if 'activation_function' in train_config:  
        activation_function = train_config['activation_function']
    else:
        activation_function = 'tanh'
    if 'learning_rate' in train_config:  
        learning_rate = train_config['learning_rate']
    else:
        learning_rate = 1e-6
    if 'decay' in train_config:  
        decay = train_config['decay']
    else:
        decay = 1e-6
    if 'epsilon' in train_config:  
        epsilon = train_config['epsilon']
    else:
        epsilon = 1e-8
    if 'layer_1' in train_config:  
        layer_1 = train_config['layer_1']
    else:
        layer_1 = 300
    if 'layer_2' in train_config:  
        layer_2 = train_config['layer_2']
    else:
        layer_2 = 100
    if 'layer_3' in train_config:  
        layer_3 = train_config['layer_3']
    else:
        layer_3 = 100
        
    if 'RAVDESS' in datasets:
        x, y = loadRAVDESS()
    if 'TESS' in datasets:
        x_data, y_data = loadTESS()
        if len(x) > 0:
            x = np.concatenate((x, x_data))
            y = np.concatenate((y, y_data))
        else:
            x = x_data
            y = y_data
    if 'User' in datasets:
        x_data, y_data = loadTESS()
        if len(x) > 0:
            x = np.concatenate((x, x_data))
            y = np.concatenate((y, y_data))
        else:
            x = x_data
            y = y_data
    if len(x) < 20:
        return "Error - Not enough data to train model"
    print(f"Searched datasets and found {len(x)} rows.")
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    y_labels = np_utils.to_categorical(y_encoded)
    x_train, x_test, y_train, y_test = train_test_split(x, y_labels, test_size=test_split, shuffle=True)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    
    cfg = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=cfg)
        
    model = Sequential()
    STEPS = len(y_train[0]) 
    model.add(Conv1D(layer_1, STEPS, input_shape=(x_train.shape[1:]), activation=activation_function))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv1D(layer_2, STEPS,  activation=activation_function))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(layer_3, activation=tf.nn.leaky_relu))
    model.add(Dropout(0.1))
    model.add(Dense(len(y_train[0]), activation='softmax'))
    
    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay, epsilon=epsilon)
    try:    
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    except:
        return None
    try: 
        history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=(x_test, y_test) 
                        )
    except:
        return None
    model.save(f"{CUR_DIR}\\models\\{model_name}")
    #save label encoding for this model and dataset
    np.save(f"{CUR_DIR}\\models\\{model_name}\\{model_name}-classes.npy", encoder.classes_)
    print(f"Model saved in \\models\\{model_name}")
    sess.close()
    return history.history

#trainModel("RAVDESS-TESS", config)