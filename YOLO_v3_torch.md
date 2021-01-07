### YOLOv3 torch 구현 코드 리뷰
Pytorch 로 YOLOv3 를 구현해 직접 만든 데이터셋을 활용해 이미지 디텍션을 시도했다. 
이 [깃헙](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)을 바탕으로 파이썬 모듈을 생성했다. 그 과정에서 중요하다고 생각하는 개념 및 코드를 공유하고자 한다. 

#### 1.Layers
YOLOv3 의 경우 Darknet-53 를 backbone으로 하며 후반에 detection 층을 추가해 feature extraction 및 detection 을 진행한다. 총 4 종류의 레이어로 구성되며, 반복 사용되어 YOLO 를 구성한다. feature extraction 을 위한 convolutional layers 와 skip connection 을 담당하는 shortcut layers, detection 단계에서 필요한 연산을 수행하는 층인 upsample 과 route 층이 필요하다. 마지막으로 3 종류의 피쳐 맵에 해당하는 3개의 앵커박스 정보를 포함하는 detection 층으로 구성된다.

- (1) convolutional layers
- (2) shortcut layers
- (3) rout layers
- (4) upsample layers
- (5) detection layers

#### 2. forward
#### 3. train 

