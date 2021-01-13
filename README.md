# SpeechEmotionRecognitionPython
 Performs audio analysis on short voice clips to guess emotion in the voice.


Installation:

	This is a python 3.7 program.

	To make life easier for whomever trying to run this program, there is a requirements files containing a list of packages.
	Run the following command in your python 3 environment (after navigating to program directory):

	$ pip install -r requirements.txt

Running the program:

	In the same console as above, run command:

	$ python gui_start.py

	Alternatively, open "gui_start.py" in your preferred IDE.

Brief Description + Usage:
	
	Speech emotion recognition:
	The purpose of this project is to predict emotion in human voice. The program provides a GUI for recording audio from a microphone and detecting emotion using a Neural Network model built using keras wrapper for tensorflow.
	The user can pick a model from a list, each trained on a specific dataset and parameters. 
	The user may choose to keep the recording of their voice for future training on custom datasets. First identify the emotion in the voice if the program misjudged the emotion and press "Save Audio clip".
	The program makes use of google's Speech To Text api to create a transcript.
	
	
	Training:
	To train a new model, navigate to the Training Screen by pressing the Gear icon at the bottom of the main menu.
	1. Select Dataset from drop down list. 
		The program comes with compressed versions of RAVDESS and TESS datasets freely available online. 
		The user can select "User" dataset which contains only the user's voice recordings. For better model performance train on "RAVDESS-User" as both user and RAVDESS datasets are combined together.
		
	2. Adjust parameters using sliders.
	3. Enter activation function to be used for the model, all lower case. List of all keras activation functions: https://keras.io/activations/
	4. Enter a unique name for your model. 
	5. Press the floppy disk icon to begin training. Training may take a long time if the Hard drive is slow and sample count is large. Training also takes a long time if the model contains many Cells and is trained on many Epochs.
		The icon will turn green during training. Blue upon end.
	6. Model is saved on the last epoch. Training on too many epochs can cause overfitting. Usually around 20-40 suffice.
		Metrics are displayed below the save icon. 
	
	You may now test out your model in the main menu. 
	
Errors:
	Tensorflow may throw errors during initialisation and sometimes resort to CPU. Running the program in a python console will display errors related to Tensorflow. Ensure tensorflow is installed properly if errors occur.
	