YOLOv2
===================
#### [paper](https://arxiv.org/pdf/1612.08242.pdf)

## Introduction
- YOLO v1의 문제점을 보완한 YOLO v2 소개 및 이를 기반으로 9,000개 이상의 물체를 탐지할 수 있는 YOLO 9000을 소개한다.
- YOLO v2는 PASCAL VOC 및 COCO와 같은 표준 탐지 작업에서 최첨단이다.
- YOLO v2는 VOC 2007에서 67 FPS일 때 76.8 mAP이며 40 FPS일 때 78.6 mAP을 가진다. ResNet 및 SSD를 활용한 Faster RCNN과 같은 아이들을 능가하면서도 여전히 훨씬 빠르게 작동한다.
- 3개의 Chapter로 나누어 설명한다.
  - Better
  - Faster
  - Stronger

************
## 1. Better
- YOLO v1dms Fast R-CNN과 비교했을 때 localization errors가 나타난다. 더불어, region proposal-based methods에 비교했을 때 낮은 recall 값을 갖는다.   
Better Chapter에서는 정확성과 Recall 값을 높인 방법에 대해 설명한다.


### 1-1) Batch Normalization
- YOLO의 모든 Convolutional layer에 Batch Normalization을 추가함으로써 mAP가 2% 이상 개선된다.
>   > Batch Normalization 설명 참조 : [Time Traveler](https://89douner.tistory.com/44?category=868069)


### 1-2) High Resolution Classifier
- 원래 YOLO v1은 224x224로 훈련시킨 모델을 사용하지만 detection을 위해 Input Image size를 448x448로 키운다.   
그래서 YOLO v1은 448x448 Image의 detection이 잘 되지 않으며 성능이 저하된다.
- YOLO v2는 detection 전에 Classification Network에 448x448 고해상도로 fine-tuning해서 4%의 mAP증가로 해결한다.   
  - _YOLO v1과 달리 Darknet-19모델을 기반으로 사용했기에 가능_


### 1-3) Convolutional With Anchor Boxes
``` 
Faster RCNN의 RPN은 예측계층에 Convolution layer만 사용해 직접 좌표를 예측하지 않으며 이전에 선택했던 Anchor Box를 이용해서 
Bounding Box를 예측한다. (Anchor Box에 대한 offset, 신뢰도를 예측) 좌표대신 offset을 예측하면 문제가 단순화되고 
Network가 더 쉽게 학습할 수 있다.
```
 - YOLO v2는 YOLO v1에서 Fully Connected layer 제거 후 Convolution layer를 사용한다.
 - Anchor Box 도입하여 Bounding Box를 예측한다.


> YOLO v1은 7x7인 cell의 크기가 작아 저해상도 상태로 detection하는 것과 같다. 또한, YOLO v1은 7x7x2=98 개의 적은 양의 Bounding Box로 Recall값이 낮다.
- Convolution Network의 출력을 더 높은 해상도로 만들기 위해 하나의 Pooling layer를 제거한다.
- Anchor Box를 기존 보다 3개 더 많은 5개로 설정 해준다.
- 7x7에서 13x13으로 변경해 Recall 값이 향상한다.
> Output Feature map이 13x13인 이유 :   
  Network를 축소해 448x448이 아닌 416x416 Input Image로 작동시킨다.   
  이는 물체가 이미지의 중심을 차지하는 경향이 있으므로 홀수x홀수로 Output Feature map을 설정하는 것이 좋기 때문이다.   
  그래서 416x416으로 Image를 입력하여 13x13 Output Feature map을 얻는다.
  
- YOLO v2는 Anchor Box를 활용한 Object예측은 ground truth와의 proposed box의 IOU를 통해 예측한다.
- Class예측은 물체가 있는 경우 해당 class의 조건부 확률을 예측한다.

```
Anchor Box 사용시 정확도는 낮아진다고 한다.
Anchor Box 사용하지 않은 모델은 Recall 81%인 69.5mAP 이며
Anchor Box 사용한 모델은 Recall 88%인 69.2mAP를 얻는다.
mAP은 감소했지만 Recall의 증가는 모델이 개선할 여지가 많다는 것을 의미한다.
```

### 1-4) Dimension Clusters
- 좋은 Anchor Box를 선정하기 위해 (Train dataset에 있는) ground truth bounding box에 K-means Clustering 방법을 사용해서 최적의 Anchor Box를 찾고자 한다.
  - Anchor Box를 도입하면서 2가지의 문제점이 발생한다.   
  첫째, Box dimensions을 hand picked 된다는 것이다. (다른 하나는 Direct location prediction에서 다룬다.)   
  이를 Network가 학습해서 할 수 있지만, 사전에 좋은 Anchor Box를 선정한다면 object detection이 더 잘되기에 K-means Clustering 활용한다.
- K-means는 원래 Euclidean distance를 활용한다. 하지만, YOLO에서는 유클리드 사용 시 큰 box에서 error가 많이 발생한다.
>   > Error 설명 참조 : [Time Traveler-Dimension clusters](https://89douner.tistory.com/93)
- Box 크기와 무관한 IOU를 적용한 distance-metric을 제안하여 더 좋은 Anchor Box를 추출한다.
- 연구에 따라 Anchor Box는 5개로 설정하는것이 좋은 결과라 한다.

![dimension cluster](https://user-images.githubusercontent.com/68367334/104083145-9b043300-527f-11eb-83fc-a660465348b2.png)



### 1-5) Direct location prediction
- Anchor Box를 사용하면서 발생하는 두 번째 문제점은 모델 불안정성이다. 이 불안정성은 Box의 좌표를 구할 때 많이 발생한다.
  - 이를 해결하기 위해 logistic activation을 사용해서 offset의 값을 [0, 1] 로 제한을 둔다.
- Network는 Output Feature map의 각 cell별 5개의 Bounding Box를 선정 후 각 Bounding Box의 tx, ty, tw, th, to를 예측한다.
- 예측한 좌표들로 최종 objectness와 Box에 대한 요소를 아래와 같이 구한다.

<img src='https://user-images.githubusercontent.com/68367334/104083420-bbcd8800-5281-11eb-8762-9bd82fcba92f.png' width="60%" height="60%">

```
즉, 모델의 불안정성을 해결하고 localization error를 낮추기 위해 Anchor Box를 활용해 최종 Box들을 선택 후 Bounding Box regression을 한다.
```

### 1-6) Fine-Grained Features
- YOLO v2에서는 passthrough layer를 추가해서 고해상도 특징과 저해상도 특징을 연결한다.
- 즉, 26x26x512 Feature map을 13x13x2048 Feature map과 합쳐서 13x13x3072 크기의 Feature map을 만든다.
- 이로 인해 큰 물체 뿐만 아니라 작은 물체를 localizing하는데 세밀한 기능 이점을 얻는다.
- 성능이 1% 정도 향상한다.

![yolo v2 architecture](https://user-images.githubusercontent.com/68367334/104083109-3c3eb980-527f-11eb-8eeb-9aca707087da.png)


### 1-7) Multi-Scale Training
- YOLO v2는 Fully connected layer 제거 후 Fully convolution layer와 pooling layer만 사용하기때문에 다양하게 Image 크기를 조정 가능하다.
- 10 batche 마다 Network는 32픽셀 간격으로 새 Image 크기를 선택한다.
- 이로 인해 Network가 다양한 Input dimension에 걸쳐 잘 예측하며 이는 동일한 Network가 다른 해상도에서 detection을 할 수 있다는 의미이다.


************
## 2. Faster
- Faster Chapter는 YOLO v2의 속도부분 개선에 대해 설명한다.
- YOLO v1은 VGG 혹은 GoogleNet을 기반으로 Network를 사용해 불필요한 연산이 많고 복잡하여 무겁다.
> VGG-16의 convolution layer는 224 × 224 해상도로 단일 영상에서 단일 패스에 대해 306억 9000만 부동 소수점 연산을 요구한다.   
GoogleNet은 전진 패스에 85억 2천만 개의 작업만 사용한다.

### 2-1) Darknet-19
```
Darknet-19는 19개의 convolution layer와 5개의 최대 pooling layer를 가진다.
```
- YOLO v2는 Darknet-19를 사용한다.
- Fully connected layer를 제거하였고 대부분 convolution layer 사용으로 경량화해 속도 측면에서 개선하였다.
- Darknet-19는 Image 처리하는데 55억 8천만번의 작업만 필요로 하지만 ImageNet에서 72.9%의 상위 1개 정확도와 91.2%의 상위 5개 정확도를 달성한다.

![darknet-19](https://user-images.githubusercontent.com/68367334/104084013-aa3aaf00-5286-11eb-9d31-627588a51887.png)

************
## 3. Stronger
- YOLO v2(YOLO 9000)은 9000개의 object를 detection 할 수 있다.
- 이 Chapter는 이를 위해 detection dataset과 classification dataset을 합친 방법을 설명한다.

### 3-1) Hierarchical classification
- WordNet을 기반으로 COCO dataset(Detection dataset)과 ImageNet dataset(Classification dataset)을 합쳐 계층적 구조인 WordTree 생성한다.

![wordtree](https://user-images.githubusercontent.com/68367334/104084083-8f1c6f00-5287-11eb-8674-3ea1d9008069.png)

- ImageNet을 WordTree로 바꾸어 전체에 softmax를 적용하는 것이 아닌 같은 계층끼리 묶어서 계층별 softmax를 적용해 확률을 구한다.
- 이로 인해 90.4% Top-5 accuracy를 보인다.

![wordtree softmax](https://user-images.githubusercontent.com/68367334/104084100-d4d93780-5287-11eb-9055-04dce6946296.png)


### 3-2) Joint classification and detection
- COCO detection dataset과 Full ImageNet에서 상위 9000개 class를 가져와 새로운 dataset을 만든다.
- 또한, ImageNet detection challenge dataset도 사용한다.

************
## 결론
- YOLO v2는 다른 모델들에 비해 정확도와 속도가 향상한 것을 알 수 있다.

![figure 4  yolo v2](https://user-images.githubusercontent.com/68367334/104083662-cf79ee00-5283-11eb-8bbb-87a24e5acd74.png)




************
###### _*사진 출처 : https://www.researchgate.net/figure/The-architecture-of-YOLOv2_fig4_336177198*_
###### _*사진 출처 : J. Redmon et al, YOLO9000: Better, Faster, Stronger, arXiv 1612.08242*_
