YOLOv2
===================
#### [paper](https://arxiv.org/pdf/1612.08242.pdf)

## Introduction
- YOLO v1의 문제점을 보완한 YOLO v2 소개 및 이를 기반으로 9000개 이상의 물체를 탐지할 수 잇는 YOLO 9000을 소개한다.
- YOLO v2는 PASCAL VOC 및 COCO와 같은 표준 탐지 작업에서 최첨단이다.
- YOLO v2는 VOC 2007에서 67 FPS일때 76.8 mAP이며 40 FPS일때 78.6 mAP를 가진다. ResNet 및 SSD를 활용한 Faster RCNN과 같은 아이들을 능가하면서도 여전히 훨씬 빠르게 작동한다.
- 3개의 Chapter로 나누어 설명한다.
  - Better
  - Faster
  - Stronger

************
### 1. Better
- YOLO v1dms Fast R-CNN과 비교햇을 때 localization errors가 나타난다. 더불어, region proposal-based methods에 비교했을 때 낮은 recall 값을 갖는다.   
Better Chapter에서는 정확성과 Recall 값을 높은 방법에 대해 설명한다.


#### 1) Batch Normalization
- YOLO의 모든 Convolutional layer에 Batch Normalization을 추가함으로써 mAP가 2% 이상 개선된다.
>   > Batch Normalization 설명 참조 : [Time Traveler](https://89douner.tistory.com/44?category=868069)


#### 2) High Resolution Classifier
- 원래 YOLO v1은 224x224로 훈련시킨 모델을 사용하지만 detection을 위해 Input Image size를 448x448로 키운다.   
그래서 YOLO v1은 448x448 Image의 detection이 잘 되지 않으며 성능이 저하된다.
- YOLO v2는 detection 전에 Classification Network에 448x448 고해상도로 fine-tuning해서 4%의 mAP증가로 해결한다.   
  - _YOLO v1과 달리 Darknet-19모델을 기반으로 사용했기에 가능_


 #### 3) Convolutional With Anchor Boxes
``` 
Faster RCNN의 RPN은 예측계층에 Convolution layer만 사용해 직접 좌표를 예측하지 않으며 이전에 선택했던 Anchor Box를 이용해서 
Bounding Box를 예측한다. (Anchor Box에 대한 offset, 신뢰도를 예측) 좌표대신 offset을 예측하면 문제가 단순화되고 
Network가 더 쉽게 학습할 수 있다.
```
 - YOLO v2는 YOLO v1에서 Fully Connected layer 제거 후 Convolution layer를 사용한다.
 - Anchor Box 도입하여 Bounding Box를 예측한다.

> YOLO v1은 7x7인 cell의 크기가 작아 저해상도 상태로 detection하는 것과 같다. 또한, YOLO v1은 7x7x2=98 개의 Bounding Box로 적은 양이기에 Recall값이 낮다.
- Convolution Network의 출력을 더 높은 해상도로 만들기 위해 하나의 Pooling layer를 제거한다.
- Anchor Box를 기존 보다 3개 더 많은 5개로 설정 해준다.
- 7x7에서 13x13으로 변경을 통해 Recall 값을 향상시킨다.
> Output Feature map이 13x13인 이유 :   
  Network를 축소해 448x448이 아닌 416x416 Input Image로 작동시킨다.   
  이는 물체가 이미지의 중심을 차지하는 경향이 있기 때문에 홀수x홀수로 Output Feature map을 설정하는 것이 좋기 때문이다.   
  그래서 416x416으로 Image를 입력하여 13x13 Output Feature map을 얻는다.
  
- YOLO v2는 Anchor Box를 활용한 Object예측은 ground truth와의 proposed box의 IOU를 통해 예측한다.
- Class예측은 물체가 있는 경우 해당 class의 조건부 확률을 예측한다.

```
Anchor Box 사용시 정확도는 낮아진다고 한다.
Anchor Box 사용하지 않은 모델은 Recall 81%인 69.5mAP 이며
Anchor Box 사용한 모델은 Recall 88%인 69.2mAP를 얻는다.
mAP은 감소했지만 Recall의 증가는 모델이 개선할 여지가 많다는 것을 의미한다.
```


#### 4) Dimension Clusters
- 좋은 Anchor Box를 선정하기 위해 (Train dataset에 있는) ground truth bounding box에 K-means Clustering 방법을 사용해서 최적의 Anchor Box를 찾고자 한다.
  - Anchor Box를 도입하면서 2가지의 문제점이 발생한다.   
  첫째, Box dimensions을 hand picked 된다는 것이다. (다른 하나는 Direct location prediction에서 다룬다.)   
  이를 Network가 학습해서 할 수 있지만, 사전에 좋은 Anchor Box를 선정한다면 object detection을 더 잘 할 것이라 생각하기에 K-means Clustering 활용한다.
- K-means는 원래 Euclidean distance를 활용한다. 하지만, YOLO에서는 유클리드 사용시 큰 box에서 error가 많이 발생한다.
>   > Error 설명 참조 : [Time Traveler-Dimension clusters](https://89douner.tistory.com/93)
- Box 크기와 무관한 IOU를 적용한 distance-metric을 제안하여 더 좋은 Anchor Box를 추출한다.
- 연구에 따라 Anchor Box는 5개로 설정하는것이 좋은 결과라 한다.

![dimension cluster](https://user-images.githubusercontent.com/68367334/104083145-9b043300-527f-11eb-83fc-a660465348b2.png)


#### 5) Direct location prediction
