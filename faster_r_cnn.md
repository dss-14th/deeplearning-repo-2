# 논문  
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)


# 요약
- 기존 Fast R-CNN 속도의 bottleneck이었던 region proposal 생성을 CNN 외부의 알고리즘이 아닌, CNN 내부에 region proposal을 생성할 수 있는 network를 설계함으로써 더욱 빠르면서도 정확한 region proposal을 생성할 수 있습니다. Faster R-CNN은 기존의 Fast R-CNN에 Region proposal network을 추가한 것이 핵심이라고 볼 수 있습니다.
- RPN은 Fast R-CNN에서의 detection에 이용되어지는 region proposal을 사용하기 위해 end to end 훈련을 합니다.
(end to end : 학습시스템에서 여러 단계의 필요한 처리과정을 한번에 처리합니다. 즉, 데이터만 입력하고 원하는 목적을 학습시키는 것입니다.)
- RPN은 fully-convolutional network 로써 object bounds 와 objectness scores 를 동시에 예측 할수있게 해줍니다.
- 간단한 교대 최적화(alternating optimization)를 통해 RPN과 Fast R-CNN은 컨볼루션 기능을 공유하도록 훈련될 수 있다.
 (RPN과 Fast R-CNN이 서로 convolution feature를 공유한 상태에서 번갈아가며 학습을 진행하는 형태입니다. 논문의 마감일이 얼마 남지 않아 급하게 구현을 하다 보니 저런 비효율적인 구조가 나왔다고 ICCV 2015 튜토리얼 세션에서 솔직하게 밝힌 바가 있습니다.)

# 소개
- 이 논문에서는 알고리즘의 변화가 Fast rcnn에서의 속도 문제를 효과적으로 해결 하는 최첨단 object detection networks 와 Region Proposal Network를 소개합니다.
- Fast rcnn 에 두개의 추가 conv layer를 구성하여 RPN을 구성하였습다.
- 하나는 각 conv feature map을 짧은(256-d) feature vector 로 인코딩 하는 것이고, 다른 하나는 각 conv map position에서  다양한 스케일과 비율에 상대적인 k region proposal로써 objectness score와 regressed bound를 출력하는 것입니다.
- Faster R-CNN은 기존의 Fast R-CNN에 Region proposal network을 추가한 것이 핵심이라고 볼 수 있습니다.

<img src="https://bloglunit.files.wordpress.com/2017/05/20170525-research-seminar-google-slides-2017-05-31-16-18-49.png" width="50%" height="70%" alt="process">

# RPN
- RPN 은 모든크기의 이미지를 input으로 가져와 objectness score가 들어있는 직사각형에 객체로 출력합니다. 
- 우리는 이과정을 fully convolutional network(FCN) 으로 모델링하였습니다.
- 최종목표는 RPN의 output과 frcnn의 object detection network와 계산을 연결하는 것입니다. 
- region proposal network을 생성하기 위해, 우리는 마지막 공유 컨베이어의 컨베이어 맵 출력 위로 작은 네트워크를 넣었습니다. 이 네트워크는 입력 컨볼루션 feature map 의 nxn 과 연결되어 있습니다.
- 이 벡터는 두 개의 FCN 이고, layer(reg)와 layer(cls) 입니다.
- 다시말해 RPN은 n × n 컨볼루션에 이어서 두 개의 1 × 1 컨볼루션으로 구현됩니다. 하지만, 이처럼 단순하게 RPN을 설계할 경우 문제가 발생할 수 있습니다. 

<img src="https://bloglunit.files.wordpress.com/2017/05/20170525-research-seminar-google-slides-2017-05-31-20-01-41.png" width="50%" height="70%" alt="process">




# translation invariance
- CNN에서 translation invariance란 input의 위치가 달라져도 output이 동일한 값을 갖는것을 말한다

- 이미지에 존재하는 물체들의 크기와 비율이 다양하기 때문에 고정된 N x N 크기의 입력만으로 다양한 크기와 비율을 수용하기에는 부족함이 있습니다. 이러한 단점을 보완하기 위해 미리 정의된 여러 크기와 비율의 reference box k를 정해놓고 각각의 sliding-window 위치마다 k개의 box를 출력하도록 설계하고 이러한 방식을 anchor라고 명칭하였습니다. 즉 RPN의 출력값은, 모든 anchor 위치에 대해 각각 물체/배경을 판단하는 2k개의 classification 출력과, x,y,w,h 위치 보정값을 위한 4k개의 regression 출력을 지니게 됩니다. Feature map의 크기가 W x H라면 하나의 feature map에 총 W x H x k 개의 anchor가 존재하게 됩니다. 논문에서는 3가지의 크기(128, 256, 512)와 3가지의 비율(2:1, 1:1, 1:2)을 사용해 총 anchor k=9를 최종적으로 사용하였습니다.




# A Loss Function for Learning Region Proposals
각 앵커에 (개체인지 아닌지의) 이진 클래스 레이블을 할당한다



 ###
 학습지역 제안을 위한 손실기능
훈련 RPN을 위해, 우리는 각 앵커에 (개체인지 아닌지의) 이진 클래스 레이블을 할당한다. 우리는 (i) 가장 높은 교차-오버-유니온(IoU)을 가진 앵커/앵커 또는 (ii) 모든 지면 진실 박스와 0.7보다 높은 IoU 오버랩을 가진 앵커의 두 가지 앵커에 양성 라벨을 할당한다. 지면 진실 상자 하나가 여러 앵커에 양의 레이블을 할당할 수 있습니다. IoU 비율이 모든 실측 자료 상자에 대해 0.3보다 낮은 경우 비양성 앵커에 음의 레이블을 할당한다. 긍정적이거나 부정적이지 않은 앵커는 훈련 목표에 기여하지 않는다.
이러한 정의를 사용하면 Fast R-CNN의 다중 작업 손실에 따른 객관적 기능을 최소화할 수 있다[5]. 이미지 손실 기능은 다음과 같이 정의됩니다.

여기서, i는 미니 배치에 있는 앵커의 지수이고 pi는 앵커 i가 물체가 될 것으로 예측되는 확률이다. 지면-진실 라벨 p假i는 앵커가 양이면 1이고, 앵커가 음이면 0이다.ti는 예측 경계 상자의 4개의 매개 변수화된 좌표를 나타내는 벡터이며, t假i는 포지티브 앵커와 관련된 지면 진실 상자의 좌표이다. 분류 손실 Lcls는 두 클래스에 대한 로그 손실입니다(개체 대 개체가 아님). 회귀 손실의 경우, 우리는 Lreg(ti, t²i) = R(ti - t²i)을 사용한다. 여기서 R은 [5]에 정의된 강력한 손실 함수(α L1)이다. pøi Lreg라는 용어는 양성 앵커에 대해서만 회귀 손실이 활성화되고(pøi = 1) 그렇지 않으면 비활성화된다는 것을 의미합니다(pøi 0 0). 클래스 및 규칙 계층의 출력은 각각 {pi} 및 {ti}(으)로 구성됩니다. 두 항은 Ncls 및 Nreg와 균형 가중치 α.3으로 정규화된다.

 학습지역 제안을 위한 손실기능
훈련 RPN을 위해, 우리는 각 앵커에 (개체인지 아닌지의) 이진 클래스 레이블을 할당한다. 우리는 (i) 가장 높은 교차-오버-유니온(IoU)을 가진 앵커/앵커 또는 (ii) 모든 지면 진실 박스와 0.7보다 높은 IoU 오버랩을 가진 앵커의 두 가지 앵커에 양성 라벨을 할당한다. 지면 진실 상자 하나가 여러 앵커에 양의 레이블을 할당할 수 있습니다. IoU 비율이 모든 실측 자료 상자에 대해 0.3보다 낮은 경우 비양성 앵커에 음의 레이블을 할당한다. 긍정적이거나 부정적이지 않은 앵커는 훈련 목표에 기여하지 않는다.

여기서, i는 미니 배치에 있는 앵커의 지수이고 pi는 앵커 i가 물체가 될 것으로 예측되는 확률이다. 지반-진실 라벨 p假i는 앵커가 양이면 1이고, 앵커가 음이면 0이다. ti는 예측 경계 박스의 4개의 매개 변수화된 좌표를 나타내는 벡터이며, t假i는 포지티브 앵커와 관련된 지반-진실 박스의 좌표이다. 분류 손실 Lcls는 두 클래스에 대한 로그 손실입니다(개체 대 개체가 아님). 회귀 손실의 경우, 우리는 Lreg(ti, t²i) = R(ti - t²i)을 사용한다. 여기서 R은 [5]에 정의된 강력한 손실 함수(α L1)이다. pøi Lreg라는 용어는 양성 앵커에 대해서만 회귀 손실이 활성화되고(pøi = 1) 그렇지 않으면 비활성화된다는 것을 의미합니다(pøi 0 0). 클래스 및 규칙 계층의 출력은 각각 {pi} 및 {ti}(으)로 구성됩니다. 두 항은 Ncls 및 Nreg와 균형 가중치 α.3으로 정규화된다.

그럼에도 불구하고, 우리의 방법은 이전의 특징 지도 기반 방법과 다른 방법으로 경계 상자 회귀를 달성한다 [7, 5]. [7, 5]에서 경계 상자 회귀는 임의 크기의 영역에서 풀링된 형상에 대해 수행되며 회귀 가중치는 모든 영역 크기에 의해 공유된다. 공식에서 회귀에 사용되는 형상은 형상 맵에서 동일한 공간 크기(n × n)이다. 다양한 크기를 설명하기 위해 k bounding-box regressors 집합을 학습합니다. 각 회귀 분석기는 하나의 척도 및 하나의 가로 세로 비율을 담당하며, K 회귀 분석기는 가중치를 공유하지 않습니다. 따라서, 형상이 고정된 크기/척도임에도 불구하고 다양한 크기의 상자를 예측할 수 있다.
### 최적화
완전 컨볼루션 네트워크로 자연스럽게 구현되는 RPN[14]은 역전파 및 확률적 경사 하강(SGD)에 의해 종단간 훈련될 수 있다[12]. 우리는 이 네트워크를 훈련시키기 위해 [5]의 "이미지 중심" 샘플링 전략을 따른다. 각 미니 배치에는 많은 양의 앵커와 음의 앵커가 포함된 단일 이미지에서 발생합니다. 모든 앵커의 손실 함수에 대해 최적화할 수 있지만, 이는 앵커가 우세할 때 음의 샘플로 치우치게 됩니다. 대신, 우리는 이미지에서 256개의 앵커를 무작위로 추출하여 미니 배치의 손실 함수를 계산한다. 여기서 샘플링된 양극 및 음극 앵커는 최대 1:1의 비율을 갖는다. 이미지에 양성 검체가 128개 미만일 경우 미니 배치에 음성의 샘플을 패딩합니다.

지역제안 및 객체탐지를 위한 컨볼루션 특성 공유
지금까지 우리는 이러한 제안을 활용할 지역 기반 객체 감지 CNN을 고려하지 않고 지역 제안 생성을 위한 네트워크를 훈련시키는 방법을 설명하였다. 감지 네트워크의 경우 Fast R-CNN [5]4를 채택하고 이제 RPN과 Fast R-CNN 간에 공유되는 컨베이어를 학습하는 알고리즘을 설명한다.

독립적으로 훈련되는 RPN과 Fast R-CNN은 모두 다른 방식으로 그들의 컨베이어를 수정할 것이다. 따라서 우리는 두 개의 별도 네트워크를 학습하기 보다는 두 개의 네트워크 사이에서 컨베이어를 공유할 수 있는 기술을 개발할 필요가 있다. 이는 단순히 RPN과 Fast R-CNN을 모두 포함하는 단일 네트워크를 정의한 다음 역 전파와 함께 최적화하는 것만큼 쉽지 않다. 그 이유는 Fast R-CNN 훈련은 고정된 객체 제안에 의존하며 Fast R-CNN을 학습하면서 동시에 제안 메커니즘을 변경할지 여부가 우선 순위가 명확하지 않기 때문이다. 이 공동 최적화는 향후 작업에 흥미로운 질문이지만, 우리는 교차 최적화를 통해 공유 기능을 학습하는 실용적인 4단계 훈련 알고리즘을 개발한다.

첫 번째 단계에서는 위에서 설명한 대로 RPN을 훈련한다. 이 네트워크는 ImageNet 사전 교육된 모델과 지역 제안 작업에 대해 엔드 투 엔드 미세 조정된 모델로 초기화됩니다. 두 번째 단계에서, 우리는 1단계 RPN에 의해 생성된 제안을 사용하여 Fast R-CNN에 의한 별도의 탐지 네트워크를 훈련시킨다. 또한 이 감지 네트워크는 ImageNet 사전 교육 모델에 의해 초기화됩니다. 이 시점에서 두 네트워크는 컨베이어를 공유하지 않는다. 세 번째 단계에서는 검출기 네트워크를 사용하여 RPN 훈련을 초기화하지만 공유 컨베이어를 수정하고 RPN 고유의 레이어만 미세 조정한다. 이제 두 네트워크는 콘트롤러를 공유합니다. 마지막으로, 공유 컨베이어를 고정하여 Fast R-CNN의 fc 레이어를 미세 조정한다. 따라서 두 네트워크는 동일한 컨베이어를 공유하고 통합 네트워크를 형성한다.

### Implementation Details
For anchors, we use 3 scales with box areas of 1282, 2562, and 5122 pixels, and 3 aspect ratios of 1:1, 1:2, and 2:1. We note that our algorithm allows the use of anchor boxes that are larger than the underlying receptive field when predicting large proposals. Such predictions are not impossible— one may still roughly infer the extent of an object if only the middle of the object is visible. With this design, our solution does not need multi-scale features or multi-scale sliding windows to predict large regions
