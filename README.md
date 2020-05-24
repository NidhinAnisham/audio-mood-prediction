# Audio Mood Prediction
Predicts the mood of an audio file as party,happy,sad or romantic.

Requirements:
Python 3.7
numpy
pandas
sklearn
matplotlib
pyAudioAnalysis (pip install pydub)
eyed3 (pip install eyed3)
ffmpeg (conda install -c conda-forge ffmpeg)
pickle
seaborn

Execution:
Run "predictAudioMood.py"

Algorithm:
1.	Extract useful features from the music by pre-processing the mp3 file.
2.	Use Semi-supervised learning with k-Means clustering to segregate and label the audio files into one of the moods. 
3.  An ensemble of XGBoost and SVM models predict the mood of an audio file. All the machine learning algorithms were implemented from scratch.
4.  The trained models are stored using pickle.
5.	"predictAudioMood.py" accepts an input song and outputs predicted mood for the song.
