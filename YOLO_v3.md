### YOLOv3
#### [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


#### bounding box prediction
- yolov3 는 yolo9000과 같이 차원 클러스터링을 통해 앵커 박스로부터 바운딩 박스를 예측한다. 욜로 네트워크는 각 바운딩 박스로부터 4개의 좌표를 예측하고(tx, ty, tw, th), 각 그리드셀의 왼쪽 상단 지점을 기준으로 셀 내부의 위치를 예측해 최종 바운딩 박스 정보를 결정한다.(bx,by, bw, bh)

- yolov3 는 이진 분류를 통해 objectness score (바운딩 박스에 물체가 있는지 없는지에 대한 확률점수) 를 예측한다. ground truth box(정답으로 라벨링된 박스)와 IOU 가 가장 높은 바운딩 박스의 objectness score 는 1이 되어야 한다. 이 바운딩 박스를 제외한 objectness 가 threshold 값인 0.5를 넘는 값을 가진 바운딩 박스들은 모두 무시한다. 이전 yolo 버전과 다르게, 하나의 ground truth box 에 대해 하나의 바운딩 박스를 할당한다. ground truth box 에 할당되지 않은 바운딩 박스들은 좌표나 분류 예측예 대한 손실값을 반영하지 않는다. 

####  class prediction
- yolov2 의 경우, 클래스 분류시 소프트맥스를 사용해 하나의 대상에 대해 하나의 클래스만 대응되어, 멀티 라벨이 불가능하다는 문제가 있었다. yolov3 는 이를 해결하기 위해 시그모이드를 사용해 이진분류 문제로 바꾸면서 멀티라벨을 가능하게 했다. (ex) 여자, 사람)


#### predictions across scales
- yolov3 는 3개의 다른 scale 에서 앵커박스들을 생성하고, 바운딩 박스를 예측한다. 
"""수식"""
N : 각 scale 당 그리드셀 크기(13,26,52), 4개의 바운딩박스 좌표, 1개 objectness , 80개의 클래스 예측

52*52, 26*26, 13*13 feature map을 각각 f_52, f_26, f_13 이라고 했을 시, f_26 은 *2 업샘플링, f_13은 *4 업샘플링을 통해 f_52의 정보와 합친다. 이 방법을 통해 초기 특징맵에서 (f_52, f_26) 더 정교하고 의미있는 정보를 얻을 수 있다. 이를 통해 비슷한 tensor 로 예측할 수 있게 되었다. 

또한 미리 바운딩 박스의 갯수를 결정하기 위해 k-means clustering을 사용해 앵커박스의 갯수를 지정했다. 3개의 피처맵당 3개의 클러스터를 사용했다.(3*3=9개의 anchor box)

#### feature extractor
- yolov3 에 대해 새로운 feature extractor를 사용한다. 우리의 새로운 네트워크는 YOLOv2 구조 및 YOLOv2의 backbone 이었던 Darknet-19 와 residual network 로 구성되어있다. 연속적인 3 by 3, 1by 1 컨볼루션 레이어를 사용하며, shortcut connection을 사용해 상대적으로 커졌다. 총 53개의 컨볼루션 레이어로 구성된다.(Darknet-53)
Darknet-53 은 Darknet-19보다 강력하고, ResNet-101, ResNet-152 보다 효과적이다.

##### 1 by 1 convolutional layer






