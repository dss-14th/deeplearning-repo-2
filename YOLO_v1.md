# You Only Look Once: Unified, Real-Time Object Detection
## 1. Introduction
- 이미지를 흘끗 보고도 즉각적으로 이미지에 어떤 물체가 있고, 어디에 있고, 어떻게 상호작용 하는지 알 수 있다.
- 물체감지를 위한 빠르고 정확한 알고리즘은 컴퓨터가 특수 센서 없이도 자동차를 운전하도록 하고, 보조장치가 인간 사용자에게 실시간 장면정보를 전달할 수 있게 하며, 범용 반응로복 시스템의 가능성을 열어줄 것이다.
- 지금의 많은 탐지 시스템들은 object detection을 위해 classifier를 용도 변경 하고 있다
- DPM은 sliding window방식으로, R-CNN과 같은 최근의 접근들은 region proposal 방식으로 object detection을 수행하고자 하지만 이렇게 복합적인 파이프라인은 느리고 최적화도 어렵다.

> ```"We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities."```
- YOLO는 object detection을 이미지 픽셀에서부터 바운딩박스의 조정, 클래스의확률까지 한 번에 하는 single 회귀 문제로 reframe한다
- YOLO시스템에서 당신은 이미지에 어떤 물체가 있고 어디에 있는지 예측하기 위해 단 한번만 보면 된다.


#### Benefits over traditional method of object detection
- _**Benefit 1. YOLO is extremely fast**_
  - Detection을 회귀문제로 보기 때문에 복잡한 파이프라인이 필요 없다
  - 단순히 우리의 뉴럴 넷을 새로운 이미지에 적용하고 예측하는 것 뿐
  - TitanX GPU에서 no batch로했을 때 기본 네트워크는 45프레임/sec의속도를 갖는다. Fast version은140 fps의속도이다
  - 이 말은 즉, 스트리밍 비디오에서 실시간으로 작업이 가능함을 의미한다. 25 밀리세컨(0.025)보다 작은 지연만 있을 뿐이다
  - 게다가 YOLO는 다른 실시간 시스템에 비해 2배 이상 높은 mAP를 갖는다
 
- _**Benefit 2. YOLO resons globally about the image when making predictions**_
  - Sliding window나 region proposal 기반의 기술들과는 달리 YOLO는 training, test 동안 전체 이미지를 본다
  - 그래서 물체의 모양뿐 아니라 클래스에 대한 맥락적 정보까지 표현한다
  - 최고의 탐지 방법인 fast R-CNN 조차도 이미지 배경의 일부를 물체로 인식하는 실수를 하기도 한다. 왜냐하면 더 큰 맥락을 보지 못하기 때문이다.
  - YOLO는 배경을 잘못인식하는 에러를 fast R-CNN의 반도 안되게 줄였다.
 
- _**Benefit 3. YOLO learns generalizable representations of objects**_
  - 자연스러운 이미지를 학습하고 예술작품을 가지고 테스트할 때 YOLO는 DPM이나 R-CNN을 큰 격차로 뛰어넘는 성과를 냈다.
  - YOLO는 일반화시키는 능력이 뛰어나서 새로운 도메인이나 예상치 못한 인풋에도 심각한 오류 없이 탐지를 수행하는 것으로 보인다.
  
- YOLO는 최첨단의 detection system들에 비해서 정확도는 뒤떨어진다. 하지만 이미지에서 빠르게 물체를 찾아내고 특히 작은 물체의 위치를 정확히 표현한다.

## 2. Unified Detection
> ``` "We unify the seperate compenents of object detection into a single neural network." ```
- YOLO 네트워크는 각각의 바운딩 박스를 예측하기 위해 전체 이미지의 특징들을 이용한다.
- 이미지에 대해서는 모든 클래스의 모든 바운딩 박스를 동시에 예측한다
- YOLO의 네트워크가 이미지 전체와 이미지에 있는 모든 물체들에 대해 전역적으로 (globally) 판단하고 있음을 의미한다
- YOLO는 높은 수준의 average precision을 유지하면서도 end-to-end training을 하고 실시간의 속도를 내도록 설계되었다.

- *YOLO의 detection 방법*
  - 이미지를  S x S grid로 나눈다
  - 물체의 중심이 grid cell안에 속한다면 해당 그리드 셀은 그 물체를 탐지할 책임을 갖는다.
  - 각각의 그리드 셀은 Bounding box 와 confidence score를 예측한다. 이 신뢰도 점수는 상자에 객체가 들어 있다는 확신과 상자의 예측이 얼마나 정확한지 나타낸다.
  - 우리는 confidence를 Pr(Object) ∗ IOU <sup>truth</sup><sub>pred</sub> 물체가 있을 확률과 실제와 예측값의 IOU를곱해서 정의한다
  - 만약 셀 안에 어떤 물체도 존재하지 않는다면 confidence score는 0이 되어야 한다
  - 그렇지 않으면 신뢰 점수가 예측 상자와 실측값 사이의 IOU와 같아야 한다.
  
<Blockquote>
  
**IOU 이해하기**

<img src='https://user-images.githubusercontent.com/67793544/103612139-1ce61a80-4f67-11eb-8a17-943c1105a004.png' width="50%" height="50%">

*source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
</Blockquote>
  
- *Bounding Box의 구성*
  - x, y, z, w, h로 구성
  - x, y는 그리드 셀의 경계와 관련된 바운딩 박스의 중심을 의미한다.
  - Width, height는 전체 이미지에 비례하여 설정된다.
  - 마지막으로 confidence는 실제 물체와 예측 박스 사이의 IOU값을 나타낸다.
  - 각각의 grid cell은 C 라고 하는 class probablity를 예측하는데 이것은 그리드 셀이 물체를 포함하는지 여부에 영향을 받는다.
  - 우리는 하나의 그리드 셀에서 바운딩박스의 개수와 상관없이 한 세트의 클래스 probability만을 예측한다
  - 테스트에서 우리는 class확률과 각 개별 박스들의 confidence 예측치를 곱했다.
  - Class에속할 확률 X 물체가 있을 확률
  - 이것은 우리에게 각박스에 대해 클래스에 특화된 confidence score를 알려주었다
  - 이 스코어는 박스 안에 물체가 있을 확률과 얼마나 예측박스가 물체와 잘 맞는지를 함께 보여준다


### 2-1. Network Design

![image](https://user-images.githubusercontent.com/67793544/103612332-8cf4a080-4f67-11eb-807d-81263c08c3f1.png)
*source: Joseph Redmon et al(2016). You Only Look Once: Unified, Real-Time Object Detection.

- 우리는 이 모델을 하나의 convolutional neural network로 실행시켰고 Pascal VOC디텍션 데이터 셋에서 평가하였다.
- 첫 번째 convolutional layer는 이미지로부터 특징을 추출하고 fully connected layer는 아웃풋의 확률과 좌표를 예측한다.
- 우리의 네트워크 구조는 이미지 분류를 위한 구글넷에서 영감을 받았다.
- 우리의 네트워크는 24개의 컨볼루전 레이어와 2개의 fully connected레이어를 가진다.
- 구글넷에서 사용했던 인셉션 모듈 대신 우리는 단순하게 1x1 reduction layer를 3x3 conv. Layer 뒤에 위치 시켰다.
- Full network는 그림 3과 같다.
- 우리는 또한 빠른 물체 감지의 경계를 확장하도록 설계된 YOLO의 빠른 버전을 훈련시킨다.
- Fast YOLO는 더 적은 conv layer를사용한다 (24개가 아닌 9개) 그리고 레이어의 필터도 더 적다.
- 네트워크의 사이즈를 제외하고 training 과 test의 모든 파라미터는 YOLO 와fast YOLO가 동일하다.
- 네트워크의 마지막 output은 7x7x30 predict tensor이다.

<Blockquote>

**predict tensor의 구성**
  
- 7x7x30의 predict tensor
  - 7 x 7 : 이미지 전체를 나눈 grid cell의 숫자
  - 30 : 5/ 5/ 20으로 구분
    - 5: 그리드 셀의 첫 번째 바운딩박스의 좌표 (x, y, w, h, c) _c는 해당 바운딩박스에 물체가 있을 확률 
    - 5: 그리드 셀의 두 번째 바운딩박스의 좌표 (x, y, w, h, c) _c는 해당 바운딩박스에 물체가 있을 확률
    - 20: class의 수_ YOLO에서 사용하는 클래스가 20개인 것이 아님 (이미지넷은 1000개의 class를 가짐)
      - 여기서의 클래스의 수 20은 바운딩 박스 안의 물체가 속할 클래스의 가능성을 상위 20개만 표시한 것
      - 20개의 각 클래스별 확률은 첫 번째, 두 번째 바운딩박스 좌표의 c(물체가 해당 박스에 있을 확률)와 각각 곱해져서 최종적으로 20개의 class score를 만듦 

**predict tensor 구성: 1,2 번째 바운딩박스의 좌표구성**     
![image](https://user-images.githubusercontent.com/67793544/103613419-e067ee00-4f69-11eb-8404-a2fbe2bd489e.png)

**predict tensor 구성: 20개의 class score**
![image](https://user-images.githubusercontent.com/67793544/103614094-2d988f80-4f6b-11eb-9f6a-488cd0dc6a1e.png)

*source: https://goo.gl/eFcsTv
</Blockquote>

### 2-2. Training
- training에 사용한 데이터 셋: Imagenet 1000개 클래스
- ImageNet 2012 validation set에서 top5의 정확도가 88%를 달성했다.
- 24개의 convolutional layer 중 20개를 pretrain에 사용했다.
- detection을 위해 모델을 변경했다.
  - convolutional layer와 connected layer를 pretrain된 네트워크에 더해 성능을 높였다.
  - 네 개의 convolutional layer와 2개의 fully connected layer의 가중치를 랜덤하게 초기화하도록 하여 추가했다
- 디텍션에서는 잘 정제된 이미지가 높은 성능을 보이기 때문에 기존 google net의 인풋이었던 224x224 사이즈를 448x448로 변경했다.
- 바운딩박스의 좌표는 0과 1 사이의 값으로 표준화 했다.
- sum quare error를 사용
  - 최적화가 용이하지만 평균 precision을 최대화 하고자하는 목표와는 완벽하게 align되지 않는다.
  - **위치 에러를 분류에러와 동일한 가중치를 부여한다.**
    - 대부분의 이미지들의 많은 그리드 셀에는 물체가 없을 수 있는데, 위치에러와 분류에러에 동일한 가중치를 부여하면 물체가 있는 그리드 셀이 상대적으로 높은 점수를 갖게 될 수 있다.
    - 이로인해 트레이닝이 일찍 종료되어 모델이 불안정해질수 있다.
    - *문제 해결을 위해 바운딩박스 좌표 예측에 대한 loss를 증가시키고 물체를 포함하지 않는 박스에 대한 예측의 오차를 감소시켰다*
  - **큰 박스와 작은 박스의 에러에 모두 동일한 가중치를 부여한다.**
    - 작은 박스에서의 작은 오차가 큰 박스에서의 작은 오차보다 더 중요하다.
    - *이것을 부분적으로나마 보완하기 위해 바운딩박스의 너비와 높이에 대해서 직접값을 사용하기보다는 제곱 루트를 사용했다.*
- YOLO는 하나의 그리드셀에서 여러개의 배운딩 박스를 예측한다. (YOLO v1은 2개)
  - 이미지 상의 실제 물체를 가장 잘 탐지하는(IOU가 높은) 바운딩박스가 해당 물체를 예측하도록 했다.
  - 하나의 물체를 탐지하는 여러개의 바운딩박스가 생기지 않으면서 바운딩 박스 간에 분화가 진행되었다.
  - 각각의 예측기는 특정 사이즈, 비율, 물체의 종류 등을 더 잘 예측했고 전반적인 리콜수치는 상승했다.

- YOLO의 loss function

![image](https://user-images.githubusercontent.com/67793544/103616085-21163600-4f6f-11eb-9d65-503c1493aa1f.png)
  - ![image](https://user-images.githubusercontent.com/67793544/103616261-6c304900-4f6f-11eb-95d1-01c6f8134f72.png)는 i번째 셀에 물체가 있다는 것을 의미한다.
  - ![image](https://user-images.githubusercontent.com/67793544/103616338-97b33380-4f6f-11eb-825c-68e65de16f86.png)는 i번째 셀의 j번째 바운딩박스가 예측에 대한 책임을 갖는다 즉, j번째 바운딩 박스의 confidence score가 더 높음을 의미한다.
  - 이 loss function은 물체가 해당 그리드셀에 존재할 때만 분류 에러에 대한 패널티를 준다.
  - 예측기가 실제 박스에 대해 예측의 책임이 있을 때만 location 에러에 대해 패널티를 준다.

![image](https://user-images.githubusercontent.com/67793544/103617373-7a7f6480-4f71-11eb-9cfe-4032443d02e5.png)

*source: https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation
*source: youtube 10분딥러닝_14_YOLO https://www.youtube.com/watch?v=8DjIJc7xH5U&t=436s

- training 조건 상세
  - 135번의 epochs, PASCAL VOC 2007, 2012로 validation 했다
  - learning rate schedule
    - 첫 번째 에포크에서 우리는 학습률을 0.001에서 0.01까지 천천히 높였다.
    - 만약 우리가 우리 모델에 처음부터 높은 학습률을 적용했다면 불안정한 그라디언트때문에 모델이 수렴하지 않을 수 있다.
    - 우리는 0.01의 학습률로 75에포크를 훈련시켰고, 0.001로 30 에포크, 그리고 마지막 30번의 에포크는 0.0001의 학습률로 훈련시켰다.
  - 과적합 방지
    - dropout = 0.5로 설정했다.
    - 데이터 증강을 위해 우리는 랜덤 스캐일링과 원본 이미지 사이즈의 최대 20%까지 변형했다.
    - 우리는 또한 랜덤하게 노출을 조정하고 HSV컬러 스페이스에 최대 1.5의 팩터로 이미지의 포화를 조정했다

### 2-3. Inference
- PASCAL VOC에서 네트워크는 이미지당 98개의 배운딩 박스를 예측하고 각 박스에 대해 클래스 probability를예측한다.
- YOLO는 단 하나의 네트워크 evaluation만을 요구하기 때문에 다른 분류 기반의 방법들과는 달리 매우 빠른 속도를 보였다. 
- 대부분 물체가 들어있는 그리드셀이 명확하고 네트워크는 각각의 물체에 대해 하나의 박스만을 예측한다.
- 그러나 일부 큰 물체 또는 경계에 있는 물체들은 몇 개의 셀에 걸쳐 위치하기도 한다.
- Non-maximal suppression은 이러한 다중 디텍션을 수정하는데 사용된다.

<Blockquote>
  
**Non Maximal Suppression**
- 각각의 bounding box는 class에 속할 확률값을 가진다. (predict tensor에서 20개 클래스 중 1개)
- 해당 확률값이 0.2보다 작으면 0으로 셋팅한다.
- 7x7 grid cell에서 각 2개의 bounding box를 만들어 총 98개의 bounding box가 생성된다.
- 각 bounding box별로 강아지 class에 속할 확률, 고양이 class에 속할 확률이 정렬되면 각 클래스별로 98개의 확률값이 도출된다. (이미지 참고)

![image](https://user-images.githubusercontent.com/67793544/103620822-853cf800-4f77-11eb-844d-8851fb46fdf9.png)

*source: https://goo.gl/eFcsTv*
- 가장 높은 score를 가진 바운딩박스를 'bbox_max'로 설정하고 0은 아니지만 더 낮은 score를 가진 다른 바운딩 박스와 비교 한다. 이 바운딩 박스를 'bbox_cur'로 설정한다.
- bbox_max와 bbox_cur의 IOU값이 (교집합/합집합) 0.5를 초과하면 bbox_cur의 값을 0으로 변경한다.
- IOU값이 높다는 것은 겹치는 부분이 크다는 것이고 같은 물체를 디텍션하고 있다는 것을 의미한다.
- 다음으로 높은 bbox_cur를 선정해서 동일과정을 반복한다.
- IOU가 0.5를 넘지 않으면 수치는 그대로 두고 다음으로 넘어간다.
- 가장 높은 score를 가진 바운딩박스와의 비교가 끝나면 2번째로 큰 score를 가졌던 bouning box를 bbox_max로 설정하고 위의 과정을 다시 반복한다.
- 다시 IOU를 비교해 동일한 물체를 디텍션하는 바운딩박스를 제거한다.
- 이렇게 총 20개 클래스에 대한 바운딩 박스 소거 작업을 진행한다.

</Blockquote>

### 2-4. Limitations of YOLO
- YOLO는 각 그리드 셀에 대해 단 두개의 바운딩 박스만을 예측하도록하고 단 하나의 클래스를 갖도록 하기 위해 바운딩박스 예측에 대해 강력한 공간적 제약을 도입했다.
  - 이러한 공간적 제약은 우리모델이 예측할 수 있는 물체들의 수를 제한한다. 하나의 이미지에서 최대 49개의 물체만 디텍션이 가능하다.
  - 각 grid cell이 하나의 클래스만 예측하므로, 작은 object가 여러개 붙어있으면 제대로 예측이 어렵다.
- trianing data를 통해서만 bounding box의 형태가 학습한다.
  - 새롭거나 특이한 비율을 가진 object의 경우 정확한 예측이 어렵다
- 우리의 구조가 인풋 이미지로부터 다중 다운샘플링 레이어를 가진다.
  - 바운딩박스를 예측하는데 상대적으로 거친 특징들을 사용한다.
- 마지막으로 대략적인 디텍션 성능을 loss function으로 트레이닝 하는 동안 우리의 loss function은 작은 바운딩 박스와 큰 바운딩 박스의 에러를 동일하게 취급한다
  - 큰 박스의 작은 에러는 일반적으로 미미하지만 작은 박스에서의 작은 에러는 IOU에 훨씬 더 큰 영향을 준다. 그래서 에러의 주요 원천이 잘못된 위치정보가 된다.

## 3. Comparison to Other Detection System
- classifier, localizer 들과의 비교
- 아래 표는 논문의 내용을 정리한 표

|구분|DPM|R-CNN|Deep-multibox|Overfeat|MultiGrasp|
|:---:|:-----|:-----|:-----|:-----|:-----|
|특징|-sliding window 접근<br>-파이프라인적접근|-sliding window 대신 region proposal 사용<br>-파이프라인적 접근<br>-느린속도<br>-2000개의 바운딩박스제안|- region of interest를 찾는데 Selective search가 아닌 CNN을 사용<br>-컨피던스 예측치를 단일 클래스 예측치로 대체함으로써 단일물체 탐지 수행가능<br>-일반 물체 디텍션은 불가<br>-파이프라인적 접근, 추가적인 이미지 분류 필요|-Localization과 디텍션을 수행하기 위해 CNN 학습<br>-sliding window detection을 효율적으로 수행<br>-파이프라인적 접근(분리된시스템)<br>-위치정보를 최적화하지만 detection을 수행하지는 않음<br>-Overfeat은 전역적인 맥락을 보지 못하여 중대한 후작업을 요구함|-바운딩 박스 예측을 위한 Grid기반의 접근은 MultiGrasp의 grasps에 대한 회귀에 근간<br>-grasp의 탐지는 물체 탐지보다 훨씬 간단<br>- MultiGrasp는 단지 물체를 포함하고 있는 이미지에 대해 graspable한 단일 지역을 예측<br>-사이즈나 위치를 추정하거나 물체를 경계짓고 클래스를 예측하지 않아도 됨
|YOLO|-이질적인 부분을 단일 CNN으로 대체하여 특징 추출, 바운딩박스 예측, NPS, 맥락적 지역설정 동시 수행|-잠정적 바운딩박스 제안, conv layer를 이용한 scoring 유사<br>-공간적제약을 강화해 동일 물체에 대한 중복 탐지를 완화<br>-단 98개의 바운딩박스 제안|- 바운딩 박스의 예측을 위해 CNN을 사용하지만 YOLO는 완전한 디텍션 시스템|-단일시스템<br>-전역적 맥락에서 이미지를 파악|-YOLO는 작업의 설계에 있어서 MultiGrasp와 유사<br>-YOLO는 이미지에 있는 다양한 클래스의 많은 물체들에 대해 바운딩 박스와 class가능성을 모두 예측|

- DPM (Deformable Parts Models)
  - sliding window 접근법을 사용한다.
  - 특징 추출과 지역분류, 높은 점수의 지역을 위한 바운딩박스 예측을 위해 파이프라인을 해체시킨다(단계가 분절되어 있다)
  - YOLO는 모든 이질적인 부분을 단일 CNN으로 대체하여 특징 추출, 바운딩박스 예측, NPS, 맥락적 지역설정이 모두 동시에 수행된다.
  - 고정된 특징 대신 네트워크는 선상에 있는 특징들을 학습하고 탐지를 위해 최적화한다
  - 우리의 통합된 구조는 DPM보다 더 빠르고 더 정확한 모델을 만들었다
- R-CNN
  - R-CNN과관련 모델들은 sliding window 대신 region proposal을사용한다.
  - Selective search 는 잠정적 바운딩 박스를 만들어 내고 컨볼루전 네트워크는 특징을 추출한다. SVM이 박스에 점수를 매기는데, 선형모델로 바운딩 박스를 조정하고 NMS가 중복되는 탐지를 제거한다.
  - 이 복잡한 파이프라인의 각 단계는 독립적으로 정확하게 튜닝되어야 하고 결과 체계는 테스트 시간에서 이미지당 40초 이상을 소요할만큼 매우 느리다.
  - YOLO는 R-CNN과 각각의 그리드셀이 잠정적인 바운딩 박스를 제안한다는 것과 각각의 박스에 컨볼루져널 피처를 이용해서 점수를 매기는 것이 유사하다
  - 그러나 YOLO는 그리드 셀의 제안에 공간적 제약을 둔다. 이것은 동일한 물체에 대한 중복되는 탐지를 완화시킨다
  - YOLO는 훨씬 적은 바운딩 박스를 제안하는데, 셀렉티브 서치가 약 2000개를 제안했다면 우리는 단98개만 제안한다.
  - 마지막으로 우리의 시스템은 개별적 요소들을 하나로 결합하고 최적화한 모델이다
- Other fast detector
  - Fast, faster R-CNN은 selective search 대신 rigion proposal을사용하는 neural network를 사용하고 계산을 공유하여 R-CNN의 속도를 높이는데 중점을 두고 있다.
  - R-CNN에 비해 빨라진 속도와 정확성을 제안하지만 두 가지 모두 실시간 퍼포먼스에는 미치지 못한다.
  - 많은 연구자들은 DPM파이프라인의 속도를 높이고자 했다.
  - 거대한 디텍션 파이프라인의 개별 요소를 최적화하고자 하는 노력 대신에 YOLO는 파이프라인을 버리고 설계를 통해 빨라졌다.
  - 얼굴 또는 사람과 같은 단일 클래스에 대한 탐지는 그들이 훨씬 적은 변형을 다루기 때문에 매우 최적화되어 있다.
  - YOLO는 일반 목적의 디텍터로 다양한 물체를 동시에 탐지하는 것을 학습한다.
- Deep multibox
  - R-CNN과 달리 region of interest를 찾는것을 학습하는데 Selective search가 아닌 CNN을 사용한다.
  - Multibox는 컨피던스 예측치를 단일 클래스 예측치로 대체함으로써 단일물체 탐지 또한 수행할 수 있다.
  - 그러나 멀티박스는 일반 물체의 디텍션은 불가능하다. 그리고 여전히 거대한 디텍션 파이프라인의 일부이며, 추가적인 이미지 패치를 분류해야 한다.
  - YOLO와 멀티박스 모두 바운딩 박스의 예측을 위해 CNN을 사용하지만 YOLO는 완전한 디텍션 시스템이다.
- OverFeat
  - Localization과 디텍션을 수행하기 위한 로컬라이저의 선택을 수행하기 위해 CNN을학습한다.
  - Overfeat은 sliding window detection을 효율적으로 수행한다. 하지만 이것 또한 여전히 분리된 세스템이다.
  - Overfeat은 위치정보를 최적화하지만 detection을 수행하지는 않는다. 
  - DPM과 같이 로컬라이저는 예측을 할 때 단지 지역적 정보를 보기만 할 뿐이다.
  - Overfeat은 전역적인 맥락을 보지 못하고 일관성 있는 디텍션을 위해서 중대한 후작업을 요구한다.
- MultiGrasp
  - YOLO는 작업의 설계에 있어서 MultiGrasp와 유사하다.
  - 바운딩 박스 예측을 위한 Grid기반의 접근은 MultiGrasp의 잡을 곳(grasps)에 대한 회귀에 근간을 둔다.
  - 그러나 grasp의 탐지는 물체 탐지보다 훨씬 간단하다.
  - MultiGrasp는 단지 물체를 포함하고 있는 이미지에 대해 잡을 수 있는 단일 지역을 예측하기만 한면 된다.
  - 이것은 사이즈나 위치를 추정하거나 물체를 경계짓고 클래스를 예측하지 않아도 된다. 단지 grasping에 적합한 지역을 찾기만 하면 된다.
  - YOLO는 이미지에 있는 다양한 클래스의 많은 물체들에 대해 바운딩 박스와 class가능성을 모두 예측한다.

## 4. Experiment
- PASCAL VOC 2007을 가지고 다른 실시간 탐지 시스템들과 YOLO를 비교할 것이다.
- YOLO와 R-CNN의 변형 사이의 차이를 이해하기 위해 우리는 YOLO와 Fast R-CNN에 의해 만들어진 VOC2007의 에러를 탐색한다. 
- 다른 에러의 프로파일에 기반하여 우리는 YOLO가 Fast R-CNN 디텍션에 리소스로 사용될 수 있다는 것을 보았고 큰 성능 향상과 함께 배경을 잘못인식하는 에러를 줄이는 것을 보았다. 
- 우리는 또한 VOC 2012결과를 보여주고 mAP를 가장 최신의 모델과 비교한다.
- 결국 우리는 YOLO가 다른 탐지기들에 비해 두 개의 artwork dataset에 대해서 새로운 도메인에서 더 잘 일반화한다는 것을 확인했다.

### 4-1. 다른 실시간 시스템과의 비교
- 많은 연구들이 표준적인 탐색의 파이프라인을 빠르게 하고자 노력했다.
- 그러나 Sadeghi 외의 연구자들만이 실질적으로 실시간으로 돌아가는 디텍션시스템을 만들었다 (30 fps or better)
- 우리는 YOLO를 30hz 또는 100hz에서 구동되는 그들의 DPM이실행되는 GPU에서 비교한다.
- 물체 탐지 시스템에서 이용 가능한 정확성과 성능의 tradeoff를 설명하기 위해 다른 모델들의 상대적인 mAP와 속도를 비교할 것이다. 

<img src='https://user-images.githubusercontent.com/67793544/103632670-02706900-4f88-11eb-9fdc-8a3adc68726d.png' width="40%" height="40%">

*source: Joseph Redmon et al(2016). You Only Look Once: Unified, Real-Time Object Detection.

- Fast YOLO는 PASCAL에 대해 우리가 아는 한 가장 빠른 디텍션 방법이다. 이것은 현존하는 가장 빠른 물체 디텍터이다.
- 52.7%의 mAP를 갖는데 이것은 이전의 실시간 탐지기의 두 배가 넘는 정확성이다.
- YOLO는 실시간 성능을 유지하며 mAP를63.4% 까지높였다
- 우리는 또한 VGG16을 이용해서 YOLO를학습시켰다. 이 모델은 더 정확하지만 YOLO보다 매우 느리다.
- Fastest DPM은 큰 mAP의손실 없이 효과적으로 DPM의속도를 높였다
- 그러나 여전히 2가지 요소에서 실시간 퍼포먼스를 놓치고 있다
- 이것은 또한 neural network접근에 비해 DPM의 상대적으로 낮은 디텍션 정확도에 의해 제한된다
- R-CNN Minus R은 셀렉티브 서치를 고정된 바운딩박스 제안으로 변경했다. R-CNN보다 훨씬 빨라졌지만 여전히 실시간에는 미치지 못했고 고정된바운딩 박스가 좋은 제안을 하지 못함으로써 큰 정확도의 손실을 가져왔다
- Fast R-CNN 은 R-CNN의 분류 단계에서 속도를 높였지만 여전히 셀렉티브 서치에 의존하고 있으며 이것은 이미지에서 바운딩 박스를 만드는데 2초정도의 시간이 소요된다. 그래서 이것은 높은 mAP를 갖지만 매우느리다
- 최근의 Faster R-CNN은 셀렉티브 서치를 neural network로 대체하였다. 우리의 테스트에서 가장 정확한 모델은 7fps를 달성했고, 덜 정확하고 작은 모델(ZF)은 18fps를달성했다
- Faster R-CNN의 VGG-16version은 YOLO보다 10mAP가 더 높았지만 6배나 느렸다
- Zeiler-Fergus Faster R-CNN은 YOLO보다 단 2.5배 느렸지만 더 낮은 정확도를 보였다.

### 4-2. VOC 2007에러 분석
- YOLO와 최신 디텍터들의 차이에 대한 더 깊은 연구를 위해 우리는 VOC2007의 결과를 구체적으로 뜯어봤다.
- 우리는 YOLO를 fast R-CNN과 비교했는데, fast R-CNN이 PASCAL dataset에 대해 가장 최상의 퍼포먼스를 내는 디텍터 중 하나이고 그 디텍션을 공개적으로 이용 가능하기 때문이다
- 우리는 Hoiem et al.이 이용했던 방법론을 사용한다 (Diagnosing Error in Object Detectors)
- 테스트 시, 해당 카테고리에 대해 top N개의 예측을 살펴보고 각 예측은 정확하거나 아래의 오류 유형에 따라 분류됨
  - Correct: correct class ans IOU > .5 (올바른 클래스이고 IOU가 0.5 초과이면 correct)
  - Localization: correct class, .1< IOU < .5 (올바른 클래스이고, IOU가 0.1 초과, 0.5 미만이면 localizaion)
  - Similar: class is similar, IOU > .1 (클래스가 실제물체와 유사하고 IOU가 0.1 초과이면 Similar)
  - Other: class is wrong, IOU > .1 (클래스가 틀렸으나 IOU가 0.1 초과이면 other)
  - Background: IOU < .1 for any object (어떤 object이던간에 IOU가 1보다 작으면 background)
  
![image](https://user-images.githubusercontent.com/67793544/103633551-3bf5a400-4f89-11eb-801e-f2a46bd9d424.png)

*source: Joseph Redmon et al(2016). You Only Look Once: Unified, Real-Time Object Detection.

- 모든 20개 클래스에 대해 평균 에러타임을 각각 보여준다
- YOLO는 물체의 위치를 올바르게 알아내기 위해 애썼다. 위치 에러는 모든 다른 소스들의 결합한 것보다 YOLO의에러가 더 큰 것을 설명한다
- Fast R-CNN 훨씬 더 작은 위치 에러를 보였지만 훨씬 더 큰 배경 에러를 나타냈다
- 상위 탐지 항목 중 13.6%는 개체를 포함하지 않는 잘못된 예측이다. Fast R-CNN은 YOLO보다 배경 탐지를 거의 3배 더 예측한다.

### 4-3. Fast R-CNN과 YOLO의 결합
- YOLO는 Fast R-CNN보다 훨씬 더 적은 배경 에러를 발생시킨다.
- Fast R-CNN에서 YOLO를 사용함으로써 배경 디텍션을 제거해서 큰 성능 향상을 얻을 수 있다.
- R-CNN이 예측하는 모든 바운딩박스에서 대해 우리는 YOLO가 유사한 박스를 예측했는지를 확인할 수 있다. 만약 그렇다면 우리는 그 예측에 YOLO의 예측된 가능성을 기반으로 가중치를 줄 수 있다. 그리고 두 개의 박스를 겹친다
<img src='https://user-images.githubusercontent.com/67793544/103633982-ce964300-4f89-11eb-8b13-290a95186b5a.png' width="50%" height="50%">

*source: Joseph Redmon et al(2016). You Only Look Once: Unified, Real-Time Object Detection.

- Fast R-CNN의베스트 모델은 VOC 2007에서 mAP가 71.8%였다. YOLO와 결합했을 때 3.2% mAP가올라 75%가 되었다.
- 우리는 또 top fast R-CNN모델과 다른 버전들을 결합해보았다. 그 결과 mAP가 약간 상승했다.
- YOLO로 인한 상승은 단순히 모델간 앙상블 효과가 아니다. 다른 fast R-CNN버전과의 결합에서는 그 효과가 미미했기 때문이다. 
- 이것은 정확하게 YOLO가 test에서 다른 방법으로 오차를 만들기 때문이다. 이것이 fast R-CNN의 성능 향상에 분명히 효과적이다.
- 하지만 불행하게도 이러한 결합은 YOLO의 속도에는 도움을 주지 못했다. 각각의 모델을 돌려야 했고 결과를 결합해야했기 때문이다.

### 4-4. VOC 212 Results

![image](https://user-images.githubusercontent.com/67793544/103634171-0e5d2a80-4f8a-11eb-9c80-b4c69eb277c4.png)

- VOC2012 test set에서는 YOLO가57.9%의 mAP를 나타냈다.
- 이것은 현재의 최첨단 기술보다는 낮은 수치이고 VGG16을 사용하는 original R-CNN과 유사한 수치이다. 
- Table3을보면 우리의 시스템은 가까운 다른 경쟁자들에 비해 작은 물체 탐지에 어려움을 겪고 있다.
- 병이나 배, tv, 모니터와 같은 카테고리에서 YOLO는 R-CNN보다 8-10% 정도 낮은 mAP를 나타낸다.
- 하지만 고양이나 기차와 같은 카테고리에 대해서 YOLO는 더 높은 성능을 보였다.
- 우리가 만든 Fast R-CNN과 YOLO의 결합 모델은 디텍션 방법에서 가장 높은 성과를 냈다.
- Fast R-CNN은 YOLO와 결합하여 2.3%의 성능을 향상시키며 공개 리더보드에서 5계단 상승하였다.

### 4-5. Generalizability: Person Detenction In Artwork
- 사물 탐지를 위한 학문적인 데이터셋은 동일한 배포자에 의해 training과 test dataset을 배포받게 된다.
- 실제 적용에서는 가능한 모든 사용 사례를 예측하기가 어려우며 테스트 데이터는 시스템이 이전에 살펴본 것과 다를 수 있다.
- 우리는 예술작품에서의 사람 디텍션을 테스트하기 위한 두 데이터셋으로 Picasso dataset과 people-art dataset에 대해 다른 디텍션 시스템과 YOLO를 비교했다.  
- 5번 그림은 욜로와 다른 디텍션 방법의 성능 비교를 보여준다

<img src='https://user-images.githubusercontent.com/67793544/103728977-b1618300-5022-11eb-89ca-5dd0efd91c11.png' width="70%" height="70%">
<img src='https://user-images.githubusercontent.com/67793544/103729096-fdacc300-5022-11eb-8ce0-1628056e4137.png' width="70%" height="70%">

*source: Joseph Redmon et al(2016). You Only Look Once: Unified, Real-Time Object Detection.

- R-CNN은 VOC2007에 대해 높은 AP를 갖는다.
- 그러나 R-CNN은 artwork에 적용했을 때 상당히 많이 떨어진다.
- R-CNN은 바운딩박스 제안에 selective search를사용하는데 이것이 원본 이미지를 변형시킨다.
- R-CNN의 분류 단계에서는 단순히 아주 작은 지역만을 보며, 좋은 제안이 필요할 뿐이다.
- DPM은 artwork에 적용할 때도 AP를 꽤 잘 유지한다.
- 이전의 이론적 업적들은 DPM이 물체의 쉐입과 레이아웃에 대한 강력한 공간적 모델을 갖기 때문에 성능이 좋았음을 증명한다.
- DPM은 R-CNN만큼 많이 점수가 떨어지지는 않지만 애초에 낮은 AP에서 시작한다.
- YOLO는 VOC2007에서 좋은 성능을 보였고 artwork에 적용했을 때의 AP도 다른 방법들에 비해 덜 떨어졌다.
- DPM과같이 YOLO는 일반적으로 물체가 나타나는 위치와 물체간의 관계 뿐 아니라 물체의 모양과 사이즈까지 견본으로 만든다.
- 예술작품과 일반 이미지들은 픽셀의 수준이 매우 다르지만 물체의 사이즈나 모양은 유사하다. 그래서 YOLO는 여전히 좋은 바운딩박스를 그리고 탐지할 수 있는 것이다

## 5. Real-Tiem Detection In the World
- YOLO는 빠르고 정확한 물체 탐지기이며, 이것은 컴퓨터 비전의 적용에 적합하다.
- 우리는 YOLO를 웹캠과 연결했고 카메라에서 이미지를 가지고 오는 시간과 탐지를 보여주는 시간을 포함해서 실시간 퍼포먼스를 유지하는지 확인했다.
- YOLO는 이미지를 개별적으로 처리하지만 웹캠과 연결하면 tracking시스템처럼 작동하며, 물체가 움직이면서 외관이 변하는 것을 감지한다.

## 6. Conclusion
- 우리는 물체 탐지의 통합된 모델로서 YOLO를 소개했다.
- 우리의 모델은 만들기 단순하고 전체 이미지를 바로 학습한다.
- 분류 기반의 접근과는 다르게 YOLO는 탐지 성능과 다이렉트로 상응하는 로스펑션을 가지고 학습하게 되며, 전체 모델은 통합적으로 train된다.
- Fast YOLO는 문헌에서는 가장 빠른 범용 물체 탐지기이며,  YOLO는 실시간 객체 탐지에서 최첨단 기술을 적용한다.
- 또한 YOLO는 새로운 도메인에도 잘 일반화되어 빠르고 강력한 객체 탐지에 의존하는 애플리케이션에 이상적이다. 


