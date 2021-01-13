import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

import train

MODEL_FILENAME = "SpeechEmotionRecog-RAVDESS-TESS"
MODEL_DIR = f"{os.path.dirname(os.path.abspath(__file__))}\\models"

#sreModel = keras.models.load_model(f"{MODEL_DIR}\\{MODEL_FILENAME}")

def predict(model_name, filename):
    #load keras model
    sreModel = keras.models.load_model(f"{MODEL_DIR}\\{model_name}")
    
    data = train.extractFeature(filename)
    #reshape to a single sample
    data = np.expand_dims(data, axis=1)
    data = np.reshape(data, (1, data.shape[0], data.shape[1]))
    
    try:   
        prediction = sreModel.predict(data, verbose=1)
    except:
        return None
    # print(prediction)
    #load classes
    encoder = LabelEncoder()
    encoder.classes_ = np.load(f"{MODEL_DIR}\\{model_name}\\{model_name}-classes.npy")
    # print("classes")
    # print(encoder.classes_)
    # print("inverse transform prediction")
    #find strongest prediction index
    bestGuess = max(range(len(prediction[0])), key=prediction[0].__getitem__)
    bestGuessConfidence = prediction[0][bestGuess]
    prediction[0][bestGuess]  = 0
    secondBestGuess = max(range(len(prediction[0])), key=prediction[0].__getitem__)
    secondGuessConfidence = prediction[0][secondBestGuess]
    prediction[0][bestGuess]  = bestGuessConfidence

    # print(f"Model predicts {encoder.classes_[bestGuess]} emotion with confidence of {bestGuessConfidence*100:.1f}%")
    # print(f"Model predicts {encoder.classes_[secondBestGuess]} emotion with confidence of {secondGuessConfidence*100:.1f}%")
    return encoder.classes_[bestGuess], bestGuessConfidence, encoder.classes_[secondBestGuess], secondGuessConfidence