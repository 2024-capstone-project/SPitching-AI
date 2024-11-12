# Interview Buster

## Introduction

### Motivation

In the competitive job market, effective communication, both verbal and non-verbal, is crucial. Up to 50% of communication relies on body language. Job seekers often struggle with conveying the right non-verbal signals during interviews, impacting their chances of securing employment. This project aims to address this challenge and provide a solution that empowers job seekers to enhance their body language skills.

<img src="[https://visitcount.itsvg.in/api?id=AtulkrishnanMU&icon=0&color=0](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Mehrabian.png/640px-Mehrabian.png)" alt="Rule" />

📊 **Statistics** underscore the importance of body language in job interviews. Effective non-verbal communication sends positive messages to interviewers, revealing crucial aspects such as confidence, sincerity, and enthusiasm. Elements like posture, facial expressions, and gestures play a pivotal role in shaping interviewers’ perceptions of a candidate’s suitability for a job.

The frequency and scale of this issue are widespread, affecting job seekers across various demographics. The urgency of addressing this problem is underscored by the competitive nature of the job market and the impact body language has on interview outcomes. The call to action is to equip job seekers with the tools they need to master the art of non-verbal communication and enhance their prospects in the job market.

### Problem Statement

By leveraging MediaPipe, a deep learning framework, this project aims to develop a system that analyzes and provides feedback on various aspects of non-verbal communication. Through facial landmark detection, head pose estimation, eye contact analysis, smile detection, hand gestures, and body pose classification, this system seeks to empower job seekers with insights into their non-verbal behavior during interviews. Ultimately, the goal is to assist them in refining their body language, fostering confidence, and increasing their chances of success in the competitive job market.

## Proposed Solution

🤖 The proposed solution is an **AI-Powered Non-Verbal Communication Coach** that uses advanced computer vision techniques. For estimating the position and orientation of the head, a PnP (Perspective n Point) solver is used to obtain the roll, pitch, and yaw angles of the head. Eye contact detection involves calculating a gaze value. An Eye Aspect Ratio is calculated for blink detection. The smile detection, hand gesture classification, and body pose classification models are trained on labeled datasets of facial, hand, and pose landmark points, respectively.

## Methodology

### Overview

The proposed system integrates various components to aid job seekers in refining their non-verbal communication skills during interviews. The system includes:

1. **Facial Landmark Detection**: Using MediaPipe’s Face Mesh model to identify faces and detect 3D landmark points.
2. **Head Pose Estimation**: Utilizing a subset of facial landmarks to determine head orientation by solving the PnP problem.
3. **Eye Blink Detection**: Employing the Eye Aspect Ratio (EAR) to assess whether the eyes are open or closed.
4. **Eye Contact Detection**: Calculating gaze ratios to evaluate eye contact, considering head position.
5. **Smile Classification**: Training machine learning models on a custom dataset of facial landmarks.
6. **Hand Landmark Detection**: Utilizing MediaPipe’s Hands model to detect hand landmarks.
7. **Hand Gesture Classification**: Classifying hand gestures based on normalized and scaled landmark points.
8. **Pose Landmark Detection**: Using MediaPipe’s Pose model to detect and classify upper body poses.
9. **Feedback Generation**: Providing personalized feedback based on the analysis results.

### Components

1. **Face Landmark Detection**:
   - Utilizes MediaPipe’s Face Mesh model to detect 3D facial landmarks in each frame of the input video.
   - The model comprises the Face Detector (BlazeFace) and the 3D Face Landmark Model (based on ResNet architecture).

2. **Head Pose Estimation**:
   - Estimates head position using specific facial landmarks.
   - Employs a camera matrix to transform 3D coordinates into 2D representations.
   - Uses the Perspective n Point (PnP) algorithm to find rotation and translation vectors.

3. **Eye Blink Detection**:
   - Calculates the Eye Aspect Ratio (EAR) using six landmark points around the eye.
   - Determines the eye state (open or closed) based on the EAR threshold.

4. **Eye Contact Detection**:
   - Calculates gaze ratios for each eye.
   - Determines eye contact based on the gaze ratios and defined threshold values.
   - Uses a variable gaze value for improved accuracy.

5. **Smile Classification**:
   - Uses a custom CSV dataset of facial landmarks labeled with different types of smiles.
   - Normalizes and scales landmarks to train machine learning models for smile classification.

6. **Hand Landmark Detection**:
   - Utilizes MediaPipe’s Hands model to detect hand landmarks from input frames.
   - Includes palm detection (BlazePalm) and hand landmark prediction stages.

7. **Hand Gesture Classification**:
   - Uses a CSV file dataset containing hand landmark points labeled as different hand gestures.
   - Normalizes and scales landmarks to train machine learning models for gesture classification.

8. **Pose Landmark Detection**:
   - Utilizes MediaPipe’s Pose model to detect pose landmarks in each frame of the input video.
   - Predicts 33 landmarks on the human body using a CNN architecture.

9. **Pose Classification**:
   - Uses a CSV file dataset containing pose landmark points labeled as different poses.
   - Normalizes and scales landmarks to train machine learning models for pose classification.

10. **Feedback Generation**:
    - Provides personalized feedback based on observed results.
    - Considers various cues such as smiling, maintaining eye contact, head posture, body poses, and hand gestures.

## Conclusion

🎯 This project aims to empower job seekers by enhancing their non-verbal communication skills through an AI-Powered Non-Verbal Communication Coach. By leveraging advanced computer vision techniques and deep learning models, the system provides valuable insights and feedback, helping job seekers improve their body language and increase their chances of success in the competitive job market.

---

## Access the Application

🚀 The application is available online at [Interview Buster](https://interviewbuster.streamlit.app/).

## Usage

1. Upload a video of your interview practice session.
2. The system will analyze your non-verbal communication using various computer vision techniques.
3. Receive detailed feedback on your body language, including facial expressions, head posture, eye contact, hand gestures, and body poses.
4. Use the feedback to improve your non-verbal communication skills.
