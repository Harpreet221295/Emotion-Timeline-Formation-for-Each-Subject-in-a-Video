# Emotion-Timeline-Formation-for-Each-Subject-in-a-Video

Task : In a video, for every subject(person), create a timeline(frame number and timestamp) when they appeared in the video, labelled by the emotion they displayed in each frame

Steps :
Using opencv, converted the video into stream of images.
Used python’s face recognition library(cnn model for face detection) for detecting faces in a video frame and computing face encodings from face images- for every face, frame number, TimeStamp, face bounding box locations and face embeddings are stored
Initialized a pre-trained Deep Learning face recognition model for computing emotion from face images in video frame and for every face, its emotion label is also stored as a FaceDict object
DBSCAN clustering algorithm is used on all the face embeddings and different clusters formed denotes the different subjects(persons) to whom those encodings belong.
For every subject, FaceDict objects are sorted in increasing order of their frame numbers. A window of 10 frames is to keep the emotions and for every new frame, the mode of the emotions from that window is assigned to that face in that frame. This windowing process is continued for a series of frame numbers for each subject. When a break in this series in encountered the window is reset.

Kindly follow the steps below to execute the program - 
1) Download the folder from the link specified below
2) Open terminal inside Emotion Timeline Formation for Each Subject in a Video directory.
3) cd src
4) chmod 774 install_required_modules.sh  (all the required modules are specified in src/requirement.txt) 
5) create a virtual environment
6)./install_required_modules.sh
7) python video_emotion_rgb.py -f 15 -out './output_dir' -t 0 20000 --save_embed -v '../../Diego_Luna_Interview.mp4'
 
Kindly note the following optional arguments that you can specify for experimenting with different videos with different examples -
a) [-f]  [--frames_window_length] : to specify the size of buffer used to process the emotion label of the frames
b) [-out] [--output_directory_path]: the path of directory where emotion timeline information about each subject will be stored
c) [-v] [--video_path]: the path of the video file
d) [--save_embed]: to save the face data from all frames to file system as a pickle object named 'face_data'
e) [-e] [--face_embeddings_path]: To not compute the face embeddings again and use the previously generated data, the argument accepts the path name of the file. Default is set to 'face_data' pickle object which was specified earlier
f) [-t] [--time_interval]: To spcify the time range in millisecond for which you want to create the timeline

You will find the output of the program under output_directory_path you specified with text files as subject_0, subject_1 etc.
Within each subject_i file, the format of the result is - 
emotion_text1
starting_frame_number      ending_frame_number   starting_time   ending_time 
...
emotion_text2
starting_frame_number      ending_frame_number   starting_time   ending_time

run_video_emotion_rgb_jupyter.ipynb - can be run directly in google colab so that one can use freely available gpus
video_emotion_rgb.ipynb - Complete implementation in jupyter notebook which can be run on google colab

Links :  
Emotion Recognition DNN - https://github.com/oarriaga/face_classification
Python’s face_recognition - https://pypi.org/project/face_recognition/, https://github.com/ageitgey/face_recognition
