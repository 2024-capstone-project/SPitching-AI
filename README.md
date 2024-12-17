MediaPipe라는 딥러닝 프레임워크를 활용하여 비언어적 의사소통의 다양한 측면을 분석하고 피드백을 제공한다.
얼굴 랜드마크 탐지, 머리 자세 추정, 눈 맞춤 분석, 미소 감지, 손 동작, 그리고 신체 자세 분류 등을 통해 발표연습을 하는 사람들이 비언어적 행동에 대해 통찰을 얻을 수 있도록 돕고자 한다.

### 컴포넌트
1. **Face Landmark Detection**:
MediaPipe의 Face Mesh 모델을 사용하여 입력 비디오의 각 프레임에서 3D 얼굴 랜드마크를 탐지합니다.
이 모델은 얼굴 탐지기(BlazeFace)와 3D 얼굴 랜드마크 모델(ResNet 아키텍처 기반)을 포함합니다.

2. **Head Pose Estimation**:
특정 얼굴 랜드마크를 사용하여 머리 위치를 추정합니다.
카메라 매트릭스를 사용해 3D 좌표를 2D 표현으로 변환합니다.
Perspective n Point (PnP) 알고리즘을 사용하여 회전 벡터와 변환 벡터를 찾습니다.

3. **Eye Blink Detection**:
눈 주변의 6개 랜드마크 포인트를 사용하여 Eye Aspect Ratio (EAR)를 계산합니다.
EAR 임계값을 기준으로 눈 상태(열림 또는 닫힘)를 판단합니다.

4. **Eye Contact Detection**:
각 눈에 대해 시선 비율을 계산합니다.
시선 비율과 정의된 임계값을 기반으로 눈 맞춤 여부를 결정합니다.
정확도를 높이기 위해 가변 시선 값을 사용합니다.

5. **Smile Classification**:
다양한 유형의 미소가 레이블링된 얼굴 랜드마크의 사용자 정의 CSV 데이터셋을 사용합니다.
미소 분류를 위해 랜드마크를 정규화하고 크기를 조정하여 머신러닝 모델을 훈련합니다.

6. **Hand Landmark Detection**:
MediaPipe의 Hands 모델을 사용하여 입력 프레임에서 손 랜드마크를 탐지합니다.
손바닥 탐지(BlazePalm) 및 손 랜드마크 예측 단계가 포함됩니다.

7. **Hand Gesture Classification**:
다양한 손 제스처로 레이블링된 손 랜드마크 포인트를 포함하는 CSV 파일 데이터셋을 사용합니다.
제스처 분류를 위해 랜드마크를 정규화하고 크기를 조정하여 머신러닝 모델을 훈련합니다.

8. **Pose Landmark Detection**:
MediaPipe의 Pose 모델을 사용하여 입력 비디오의 각 프레임에서 자세 랜드마크를 탐지합니다.
CNN 아키텍처를 사용하여 인체의 33개 랜드마크를 예측합니다.

9. **Pose Classification**:
다양한 자세로 레이블링된 자세 랜드마크 포인트를 포함하는 CSV 파일 데이터셋을 사용합니다.
자세 분류를 위해 랜드마크를 정규화하고 크기를 조정하여 머신러닝 모델을 훈련합니다.

10. **Feedback Generation**:
관찰된 결과를 기반으로 개인화된 피드백을 제공합니다.
미소, 눈 맞춤 유지, 머리 자세, 신체 자세 및 손 제스처와 같은 다양한 신호를 고려합니다.

### 웹사이트 연결
🚀 [Interview Buster](https://interviewbuster.streamlit.app/).

### 사용법
1. 발표 연습 세션의 비디오를 업로드하세요.
2. 시스템이 다양한 컴퓨터 비전 기술을 사용하여 비언어적 의사소통을 분석합니다.
3. 머리 자세, 눈 맞춤, 손 제스처, 신체 자세 등 본인의 바디 랭귀지에 대한 상세한 피드백을 받습니다.
4. 피드백을 활용하여 비언어적 의사소통 능력을 향상시킬 수 있습니다.
