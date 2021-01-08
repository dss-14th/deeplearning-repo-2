### YOLOv3 torch 구현 코드 리뷰
Pytorch 로 YOLOv3 를 구현해 직접 만든 데이터셋을 활용해 이미지 디텍션을 시도했다. 
이 [깃헙](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)을 바탕으로 파이썬 모듈을 생성했다. 그 과정에서 중요하다고 생각하는 개념 및 코드를 공유하고자 한다. 

#### 1.Layers
YOLOv3 의 경우 Darknet-53 를 backbone으로 하며 ResNet에서 사용하는 skip connection 을 사용한다. 
또한 마지막에 detection 층을 추가했다. 아래 총 5 종류의 레이어로 구성되어 있다.

- convolutional layers
  <img width="873" alt="스크린샷 2021-01-07 오후 5 39 34" src="https://user-images.githubusercontent.com/68367329/103871726-c6b4db00-5110-11eb-8727-ceec158b0a20.png">

- shortcut layers
  - skip connection 정보를 저장한다. 해당 레이어들간 정보 흐름을 담당한다.
  <img width="873" alt="스크린샷 2021-01-07 오후 5 42 51" src="https://user-images.githubusercontent.com/68367329/103871731-c87e9e80-5110-11eb-9237-c395bba29238.png">

- rout layers
  - detection 단계에서 필요한 레이어 정보를 저장한다. 
  <img width="873" alt="스크린샷 2021-01-07 오후 5 41 57" src="https://user-images.githubusercontent.com/68367329/103871739-c9afcb80-5110-11eb-80af-6b0e93a89bae.png">

- upsample layers
  - detection 단계에서 필요한 upsampling 연산 정보를 저장한다.
  <img width="873" alt="스크린샷 2021-01-07 오후 5 40 26" src="https://user-images.githubusercontent.com/68367329/103871734-c87e9e80-5110-11eb-922e-b56c68662d73.png">

- detection layer
  - YOLOv3 에선 총 3 개의 레이어를 사용한다. 
  - mask에 해당하는 인덱스의 anchor 정보를 가져와 해당 피쳐 맵의 앵커박스를 지정한다. 
  - 이를 통해 하나의 그리드셀당 3개의 앵커박스를 생성할 수 있게 한다.
  <img width="873" alt="스크린샷 2021-01-07 오후 5 42 17" src="https://user-images.githubusercontent.com/68367329/103871717-c4528100-5110-11eb-84a7-8079af698f56.png">
  <img width="873" alt="스크린샷 2021-01-07 오후 5 40 54" src="https://user-images.githubusercontent.com/68367329/103871736-c9173500-5110-11eb-87de-0317ffd0d217.png">
  <img width="873" alt="스크린샷 2021-01-07 오후 5 39 57" src="https://user-images.githubusercontent.com/68367329/103871728-c7e60800-5110-11eb-92a2-fe1f0e98d56e.png">


#### 2. predict_transform 함수
- 3개의 크기가 다른 피쳐 맵에서 prediction이 이뤄지기 때문에 detection layer 의 아웃풋값을 같은 scale 로 맞춰 주는것이 필요하다. 
- 이를 위해 predict_transform 함수를 사용한다.
- 한 피쳐 맵에서 grid 사이즈와 stride 는 같은 값을 가진다. ( 그리드 사이즈만큼 이동해서 anchor box 를 지정해야 하기 때문)
- anchors 의 총 갯수는 3*3=9 개이고, 하나의 그리드당 3개의 앵커박스를 가진다.
- anchors 에 있는 수치는 anchor box의 너비와 높이를 의미하며, 이를 해당 grid size 로 나눠 비율을 구한다. 

  <img width="873" alt="스크린샷 2021-01-08 오후 4 56 12" src="https://user-images.githubusercontent.com/68367329/103989260-c7ae4100-51d2-11eb-9a86-10f55529b838.png">
  <img width="873" alt="스크린샷 2021-01-08 오후 4 56 43" src="https://user-images.githubusercontent.com/68367329/103989270-cd0b8b80-51d2-11eb-8800-83f6be4eeb06.png">
  <img width="873" alt="스크린샷 2021-01-08 오후 4 57 03" src="https://user-images.githubusercontent.com/68367329/103989271-cd0b8b80-51d2-11eb-8339-8bc148868f25.png">
  <img width="873" alt="스크린샷 2021-01-08 오후 4 57 43" src="https://user-images.githubusercontent.com/68367329/103989273-cda42200-51d2-11eb-8741-c4fbfe849fea.png">


