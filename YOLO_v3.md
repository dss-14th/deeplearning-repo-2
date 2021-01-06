### YOLOv3
#### [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


#### 1. bounding box prediction
- yolov3 는 yolo9000과 같이 차원 클러스터링을 통해 앵커 박스로부터 바운딩 박스를 예측한다. 욜로 네트워크는 각 바운딩 박스로부터 4개의 좌표를 예측하고(tx, ty, tw, th), 각 그리드셀의 왼쪽 상단 지점을 기준으로 셀 내부의 위치를 예측해 최종 바운딩 박스 정보를 결정한다.(bx,by, bw, bh)


<p align="center"><img width="487" alt="스크린샷 2021-01-06 오후 1 03 47" src="https://user-images.githubusercontent.com/68367329/103729548-05209c00-5024-11eb-92bb-9f5a1da998fd.png"></p>

- yolov3 는 이진 분류를 통해 objectness score (바운딩 박스에 물체가 있는지 없는지에 대한 확률점수) 를 예측한다. ground truth box(정답으로 라벨링된 박스)와 IOU 가 가장 높은 바운딩 박스의 objectness score 는 1이 되어야 한다. 이 바운딩 박스를 제외한 objectness 가 threshold 값인 0.5를 넘는 값을 가진 바운딩 박스들은 모두 무시한다. 이전 yolo 버전과 다르게, 하나의 ground truth box 에 대해 하나의 바운딩 박스를 할당한다. ground truth box 에 할당되지 않은 바운딩 박스들은 좌표나 분류 예측예 대한 손실값을 반영하지 않는다. 

#### 2. class prediction
- yolov2 의 경우, 클래스 분류시 소프트맥스를 사용해 하나의 대상에 대해 하나의 클래스만 대응되어, 멀티 라벨이 불가능하다는 문제가 있었다. yolov3 는 이를 해결하기 위해 시그모이드를 사용해 binary cross-entropy loss를 사용하면서 멀티라벨을 가능하게 했다. (ex) 여자, 사람)


#### 3. predictions across scales
- yolov3 는 3개의 다른 scale 에서 앵커박스들을 생성하고, 바운딩 박스를 예측한다. 

<p align='center'><img width="1342" alt="스크린샷 2021-01-06 오후 1 18 52" src="https://user-images.githubusercontent.com/68367329/103729815-9c85ef00-5024-11eb-9796-0822e9d50e6d.png"></p>

코코데이터셋에서의 각 피처맵당 생성되는 정보는 다음 수식으로 계산될 수 있다.

<p align='center'><img width="198" alt="스크린샷 2021-01-06 오후 3 42 17" src="https://user-images.githubusercontent.com/68367329/103737581-ce538180-5035-11eb-8512-e17617b29c36.png"></p>

  - N : 각 scale 당 그리드셀 크기(13,26,52), 4개의 바운딩박스 좌표, 1개 objectness , 80개의 클래스 예측

52*52, 26*26, 13*13 feature map을 각각 f_52, f_26, f_13 이라고 했을 시, f_26 은 *2 업샘플링, f_13은 *4 업샘플링을 통해 f_52의 정보와 합친다. 이 방법을 통해 초기 특징맵에서 (f_52, f_26) 더 정교하고 의미있는 정보를 얻을 수 있다. 이를 통해 비슷한 tensor 로 예측할 수 있게 되었다. 

또한 미리 바운딩 박스의 갯수를 결정하기 위해 k-means clustering을 사용해 앵커박스의 갯수를 지정했다. 3개의 피처맵당 3개의 클러스터를 사용했다.(3*3=9개의 anchor box)

#### 4. yolov1, yolov2 와 비교 

<img width="1116" alt="스크린샷 2021-01-06 오후 1 20 26" src="https://user-images.githubusercontent.com/68367329/103729658-46b14700-5024-11eb-9fbc-cdc65bb26614.png">


#### 5. feature extractor
- yolov3 에 대해 새로운 feature extractor를 사용한다. 우리의 새로운 네트워크는 YOLOv2 구조 및 YOLOv2의 backbone 이었던 Darknet-19 와 residual network 로 구성되어있다. 연속적인 3 by 3, 1by 1 컨볼루션 레이어를 사용하며, shortcut connection을 사용해 상대적으로 커졌다. 총 53개의 컨볼루션 레이어로 구성된다.(Darknet-53)
Darknet-53 은 Darknet-19보다 강력하고, ResNet-101, ResNet-152 보다 효과적이다.

<p align="center"><img width="250" alt="스크린샷 2021-01-06 오후 1 04 27" src="https://user-images.githubusercontent.com/68367329/103729593-21bcd400-5024-11eb-9256-759d7fa922f0.png"></p>

##### 1 by 1 convolutional layer

맥스 풀링 단점 : 특성 맵 자체가 작아지면서 정보가 소실되어 이미지의 해상도가 낮아진다. 1 by 1 convolutional layer 의 경우 특성맵의 크기는 유지한 채, 필터의 갯수를 적게 사용해 이전 레이어보다 차원을 줄인다.이를 통해 특성 맵의 이미지 정보를 압축시킨다.

#### 6. Things We Tried That Didn't Work

아래 방법은 Yolov3 를 개발하는 동안 시도했던 시행착오들이다. 

- Anchor box x, y offset predictions 
  - 처음엔 linear activation function을 활용해 width와 height 의 곱으로 x,y offset을 예측하는 일반적인 앵커박스 예측 매커니즘을 따르려고 했다. 그러나 이 방식이 모델의 안정성을 떨어트리고, 성능도 떨어트린다는 것을 알게 되었다.
- Linear x,y predictions instead of logistic
  - logistic activation 대신 linear activation 을 사용해 바로 x,y offset 을 예측하려 했으나 mAP 가 2점 하락하는 결과를 가져왔다.

- Focal loss 
  - focal loss 를 사용하려 했으나 mAP 를 2점 떨어뜨렸다. 아마 YOLOv3가 자체 objectness 점수와 조건부 클래스 예측을 가지고 있어  focal loss 가 풀고자 하는 문제에 이미 대응할 수 있기 때문일 것이다.

- Dual IOU thresholds and truth assignment
  - Faster RCNN 은 트레이닝 단계에서 2개의 IOU threshold(0.3, 0.7)를 사용한다. 바운딩박스와 ground truth box 의 IOU 값이 0.7 이상인 경우 positive example, 0.3-0.7 사이의 경우 무시, 0.3 이하일 경우 negative example 로 간주한다. 이러한 방식을 YOLOv3 에 적용하고자 했지만 좋은 결과를 가져오지 못했다.

#### 7. result

<p align="center"><img width="606" alt="스크린샷 2021-01-06 오후 3 22 40" src="https://user-images.githubusercontent.com/68367329/103736172-1329e900-5033-11eb-8172-81a5063b896c.png"></p>
