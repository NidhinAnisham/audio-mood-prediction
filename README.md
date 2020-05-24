# Audio Mood Prediction
Predicts the mood of an audio file as party,happy,sad or romantic.

Requirements:
1.  Python 3.7
2.  numpy
3.  pandas
4.  sklearn
5.  matplotlib
6.  pyAudioAnalysis (pip install pydub)
7.  eyed3 (pip install eyed3)
8.  ffmpeg (conda install -c conda-forge ffmpeg)
9.  pickle
10.  seaborn

Execution:
Run "predictAudioMood.py"

Algorithm:
1.	Extract useful features from the music by pre-processing the mp3 file.
2.	Use Semi-supervised learning with k-Means clustering to segregate and label the audio files into one of the moods. 
3.  An ensemble of XGBoost and SVM models predict the mood of an audio file. All the machine learning algorithms were implemented from scratch.
4.  The trained models are stored using pickle.
5.	"predictAudioMood.py" accepts an input song and outputs predicted mood for the song.

File descriptions:
1.  getDataSet.py Feature Extraction from mp3 files. Output is stored in MusicData.csv
2.  getNoOfClusters.py Prints the elbow method graph for number of clusters = 1 to 10
3.  kMeans.py Peforms clustering of data. Output is stored in LabelledMusicData.csv
4.  XGBoost.py Creates the XGBoost model
5.  SVM.py Creates the SVM model
6.  getMetrics Utility function to get the different metrics of the algorithms
7.  PrintMetrics.py Prints all the different metrics of the algorithms
8.  predictAudioMood.py Predicts the mood of an audio file
