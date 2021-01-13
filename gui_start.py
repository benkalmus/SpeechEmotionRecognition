import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.config import Config
import os
import glob
import time
import threading
import speech_recognition as sr
#program python files
import recordmic
import train
import predict

kivy.require('1.11.1')  #require version
OPTIONS_FILE = "prevOptions.txt"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
validFunctions = ['elu', 'softmax', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'tanh', 'hard_sigmoid', 'sigmoid', 'exponential', 'linear']
#do not forget to remove self.clipName upon exit program

Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '1000')
Config.write()

def loadModels():
    listOfModels = []
    for folder in glob.glob(f"{CURRENT_DIR}\\models\\*"):
        listOfModels.append(os.path.basename(folder))
    return listOfModels

class StartPage(Widget):
    def __init__(self, **kwargs):
        self.buildLists()
        super(StartPage, self).__init__(**kwargs)
        #open prev options file
        if os.path.isfile(OPTIONS_FILE):
            with open(OPTIONS_FILE, "r") as fopt:
                opt = fopt.read().split(',')  #comma delimiter
    def buildLists(self):
        self.NN_MODELS = loadModels()
        self.DATASETS = ['RAVDESS-TESS', 'RAVDESS', 'TESS', 'RAVDESS-User', 'User']
        self.emotionsList = [s.capitalize() for s in train.trainingEmotions]
        self.isRecording = False
        self.clipName = "tempClip"
        
    def updateModelList(self):
        self.ids.modelDropList.values = self.NN_MODELS = loadModels()
        
    def modelChanged(self, spinner, text):
        self.modelName = self.ids.modelDropList.text
                  
    def recordBtnPress(self, instance):
        if self.isRecording:
            popupMsg = PopUp()
            popupMsg.setText("Your voice is already being recorded.")
            popupMsg.open()
            return
        #check if model is selected, show popup
        if self.ids.modelDropList.text not in self.NN_MODELS:
            popupMsg = PopUp()
            popupMsg.setText("Please select a Neural Network Model to be used for emotion guessing.")
            popupMsg.open()
            return
        
        #Change colour of Button to indicate recording
        self.ids.recordBtn.background_normal = 'icn/mic_red.png'
        self.isRecording = True
        threading.Thread(target=self.recordMic).start()
    
    def recordMic(self):        
        tempClipDirectory = f"{CURRENT_DIR}\\data\\User\\{self.clipName}.wav"
        recordingTime = round(self.ids.recordLengthSlider.value, 1)
        recordmic.recordAudio(tempClipDirectory, recordingTime)
        self.ids.recordBtn.background_normal = 'icn/mic_blue.png'
        #recording ended. Predict emotion using selected model
        bestGuess, bestGuessConfidence, secondGuess, secondGuessConfidence = predict.predict(self.ids.modelDropList.text, tempClipDirectory)
        if bestGuess == None:
            self.ids.recordResultLabel.text = "Model loading failed."
            self.isRecording = False
            self.ids.recordBtn.background_normal = 'icn/mic_green.png'
            return
        self.ids.recordResultLabel.text = f"Predicted emotion: {bestGuess.capitalize()} with confidence of {bestGuessConfidence*100:5.2f}% \nNext guess is: {secondGuess.capitalize()} with confidence of {secondGuessConfidence*100:5.2f}%"
        self.ids.emotionDropList.text = bestGuess.capitalize()
        #speech to text
        r = sr.Recognizer()
        with sr.AudioFile(tempClipDirectory) as source:
            audio = r.record(source)
            try:
                textTranscript = r.recognize_google(audio)
                if len(textTranscript) > 1:
                    textTranscript = textTranscript.capitalize()
                    self.ids.recordTranscript.text = f"Transcript: {textTranscript}"
            except sr.UnknownValueError:
                self.ids.recordTranscript.text = f"Error: Unknown Value"
            except sr.RequestError:
                self.ids.recordTranscript.text = f"Error: Service is down"
                
        #change back the color
        self.ids.recordBtn.background_normal = 'icn/mic_green.png'
        self.isRecording = False
        
    def saveClipBtnPress(self, instance):
        #retrieve predicted emotion from list
        emotion = self.ids.emotionDropList.text
        timeStr = time.strftime("%Y.%m.%d_%H.%M.%S")
        #rename temp file if exists
        if os.path.exists(f"{CURRENT_DIR}\\data\\User\\{self.clipName}.wav"):
            os.rename(f"{CURRENT_DIR}\\data\\User\\{self.clipName}.wav", f"{CURRENT_DIR}\\data\\User\\{timeStr}-{emotion.lower()}.wav")
            popupMsg = PopUp()
            popupMsg.setText("Voice sample saved.")
            popupMsg.open()
            
    
    def retrainBtnPress(self, instance):
        if not self.isRecording:
        #open training screen  
            app.screenManager.current = "Train"
    
    def sliderUpdateLabelValue(self):
        value = self.ids.recordLengthSlider.value
        self.ids.recordLengthLabel.text = f"{value:.1f} seconds"


class TrainPage(Widget):
    def __init__(self, **kwargs):
        self.buildLists()
        super(TrainPage, self).__init__(**kwargs)
        
    def buildLists(self):
        self.NN_MODELS = loadModels()
        self.DATASETS = ['RAVDESS-TESS', 'RAVDESS', 'TESS', 'RAVDESS-User', 'User']
        self.currentTime = time.strftime("%Y.%m.%d")
        #self.learningRates = [10**n *1e-8 for n in range(1, 6)]
        self.isTraining = False
        
    def returnBtnPress(self, instance):
        app.startPage.updateModelList()
        app.screenManager.current = "Start"
        
    def trainBtnPress(self):
        if self.isTraining:
            popupMsg = PopUp()
            popupMsg.setText("Please be patient as the model is being trained. Training depends on the amount of time and the speed of Hard Drive during loading of audio files.")
            popupMsg.open()
            return
        #load settings
        modelName = self.ids.modelNameInput.text.rstrip()
        dataset = self.ids.datasetTrainList.text
        testSplit = self.ids.dataSplitSlider.value
        batch = int(self.ids.batchSlider.value)
        epochs = int(round(self.ids.epochSlider.value))
        activationFunction = self.ids.activationFuncInput.text.rstrip()
        learningRate = self.ids.lrSlider.value
        layer1 = int(self.ids.layer1Slider.value)
        layer2 = int(self.ids.layer2Slider.value)
        layer3 = int(self.ids.layer3Slider.value)
        #sanitise inputs
        if modelName == "":
            popupMsg = PopUp()
            popupMsg.setText("Must enter a name for your model.")
            popupMsg.open()
            return
        modelName = f"{modelName}-{dataset}"
        if modelName in self.NN_MODELS:
            popupMsg = PopUp()
            popupMsg.setText("Please enter a unique name for your model.\nModel with identical name will overwrite previous.")
            popupMsg.open()
            return
        if dataset not in self.DATASETS:
            popupMsg = PopUp()
            popupMsg.setText("Please select a Dataset to train the network on.\nDatasets separated by hyphens will be merged before training.")
            popupMsg.open()
            return
        if activationFunction not in validFunctions:
            popupMsg = PopUp()
            popupMsg.setText("Activation function not valid. Refer to https://keras.io/activations/ for list.")
            popupMsg.open()
            return
        #build config dictionary
        self.config = {
            'dataset':dataset,
            'test_split':testSplit,
            'batch_size':batch,
            'epochs':epochs,
            'activation_function':activationFunction,
            'learning_rate':learningRate,
            'layer_1': layer1,
            'layer_2': layer2,
            'layer_3': layer3
            }
        self.ids.trainBtn.background_normal = 'icn/save_green.png'    
        self.isTraining = True
        threading.Thread(target=self.trainModel).start()
        
    def trainModel(self):  
        modelName = self.ids.modelNameInput.text.rstrip()
        dataset = self.ids.datasetTrainList.text
        modelName = f"{modelName}-{dataset}"
        epochs = int(self.ids.epochSlider.value)
        #pass config to train funciton
        history = train.trainModel(modelName, self.config)
        if history == None:
            self.ids.trainLabel.text = "Exception caught during training. Please ensure you have tensorflow and keras installed."
            self.isTraining = False
            self.ids.trainBtn.background_normal = 'icn/save_blue.png'     
            return
        
        #display validation accuracy for last epoch
        accuracy = history['categorical_accuracy'][-1]
        valAccuracy = history['val_categorical_accuracy'][-1]
        print(valAccuracy)
        self.ids.trainLabel.text = f"Training completed after {epochs} epochs: Training accuracy: {accuracy*100:.2f}%   Validation Accuracy: {valAccuracy*100:.2f}%\nModel saved at  /models/{modelName}"
        #reload list of models.
        app.startPage.updateModelList()
        self.isTraining = False
        self.ids.trainBtn.background_normal = 'icn/save_blue.png'       
        
class PopUp(Popup):
    def setText(self, text):
        self.Text = text

class SREApp(App):
    def build(self):
        self.screenManager = ScreenManager()
        self.startPage = StartPage()
        screen = Screen(name="Start")
        screen.add_widget(self.startPage)
        self.screenManager.add_widget(screen)
        
        self.trainPage = TrainPage()
        screen = Screen(name="Train")
        screen.add_widget(self.trainPage)
        self.screenManager.add_widget(screen)
        return self.screenManager
        #return StartPage()
    
    
if __name__=='__main__':
    app = SREApp()
    app.run()
    #cleanup temp files
    if os.path.exists(f"{CURRENT_DIR}\\data\\User\\tempClip.wav"):
        os.remove(f"{CURRENT_DIR}\\data\\User\\tempClip.wav") 
        
    
    