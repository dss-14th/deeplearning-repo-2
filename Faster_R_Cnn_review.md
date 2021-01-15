# Faster_R_Cnn
- [paper](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

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
- 하나는 각 conv feature map을 짧은(256-d) feature vector 로 인코딩 하는 것이고, 다른 하나는 각 conv map position에서  다양한 스케일과 비율에 상대적인 k region proposal로써 objectness score와 object bounds를 출력하는 것입니다.
- Faster R-CNN은 기존의 Fast R-CNN에 Region proposal network을 추가한 것이 핵심이라고 볼 수 있습니다.

<p align='center'><img src="https://bloglunit.files.wordpress.com/2017/05/20170525-research-seminar-google-slides-2017-05-31-16-18-49.png" width="40%" height="75%" alt="process"></p>

# RPN
- RPN 은 모든크기의 이미지를 input으로 가져와 objectness score가 들어있는 직사각형에 객체로 출력합니다. 
- 우리는 이과정을 fully convolutional network(FCN) 으로 모델링하였습니다.
- 최종목표는 RPN의 output과 Fast_r_cnn의 object detection network와 계산을 연결하는 것입니다. 
- Region proposal network을 생성하기 위해, 우리는 마지막 공유 컨베이어의 컨베이어 맵 출력 위로 작은 네트워크를 넣었습니다. 이 네트워크는 입력 컨볼루션 feature map 의 nxn 과 연결되어 있습니다.
- 이 벡터는 두 개의 FCN 이고, layer(reg)와 layer(cls) 입니다.
- 다시말해 RPN은 n × n 컨볼루션에 이어서 두 개의 1 × 1 컨볼루션으로 구현됩니다. 하지만, 이처럼 단순하게 RPN을 설계할 경우 문제가 발생할 수 있습니다. 아래에서 좀 더 자세히 설명하겠습니다. 

<p align='center'><img src="https://bloglunit.files.wordpress.com/2017/05/20170525-research-seminar-google-slides-2017-05-31-20-01-41.png" width="40%" height="75%" alt="process"></p>



# Translation invariance
- CNN에서 translation invariance란 input의 위치가 달라져도 output이 동일한 값을 갖는것을 의미합니다.

- 이미지에 존재하는 물체들의 크기와 비율이 다양하기 때문에 고정된 N x N 크기의 입력만으로 다양한 크기와 비율을 수용하기에는 부족함이 있습니다. 이러한 단점을 보완하기 위해 미리 정의된 여러 크기와 비율의 reference box k를 정해놓고 각각의 sliding-window 위치마다 k개의 box를 출력하도록 설계하고 이러한 방식을 anchor라고 명칭하였습니다. 즉 RPN의 출력값은, 모든 anchor 위치에 대해 각각 물체/배경을 판단하는 2k개의 classification 출력과, x,y,w,h 위치 보정값을 위한 4k개의 regression 출력을 지니게 됩니다. Feature map의 크기가 W x H라면 하나의 feature map에 총 W x H x k 개의 anchor가 존재하게 됩니다. 논문에서는 3가지의 크기(128, 256, 512)와 3가지의 비율(2:1, 1:1, 1:2)을 사용해 총 anchor k=9를 최종적으로 사용하였습니다.

<p align='center'><img src="https://bloglunit.files.wordpress.com/2017/06/1506-01497-pdf-2017-06-01-19-30-11.png" width="40%" height="75%" alt="process"></p>

<p align='center'><img src="https://user-images.githubusercontent.com/66449614/103753540-80974300-504e-11eb-92f1-71738b3cf581.png"></p>

# A Loss Function for Learning Region Proposals
- 각각 앵커는 물체인지 아닌지를 판단하는 이진 클래스 레이블을 가지게 됩니다. 우리는 2가지의 앵커 박스에 양성 라벨을 할당하는데, 첫째는 ground truth 박스와 겹쳐져서 가장 높은 Intersection over Union(IoU)을 가진 앵커박스이고, 두번째는 any ground truth 박스와 겹쳐져서 0.7보다 높은 IoU를 가진 앵커박스입니다. 이렇게 된다면 하나의 ground truth 박스에 여러개의 앵커 박스들이 양성 라벨을 할당 받을 수 있게 됩니다. IoU 비율이 모든 ground truth 상자에 대해 0.3보다 낮은 경우 음성 label을 할당 받습니다. 양성 또는 음성 label을 가지지 않은 앵커는 훈련 대상에서 제외됩니다.
 [IOU참고](https://ballentain.tistory.com/12)
- 이러한 정의를 사용하면 Fast RCNN의 multi-task loss 를 최소화 할 수 있습니다. 이미지에 대한 손실 함수는 다음과 같이 정의됩니다.
<p align='center'><img src="https://user-images.githubusercontent.com/66449614/103753542-81c87000-504e-11eb-974f-d724b59a4eb8.png"></p>

- 여기서 i는 미니 배치안에 있는 앵커 index이고 pi는 앵커i가 물체가 될 것으로 예측되는 확률입니다. ground-truth label 인 (p∗i)는 앵커가 양이면 1이고, 앵커가 음이면 0입니다. ti는 bounding box로 예측되어진 4개의 변수화된 좌표를 나타내는 벡터이며, (t*i)는 양성 라벨을 가진 앵커와 관련된 ground truth 상자의 좌표입니다. Lcls는 두 클래스(object vs. not object)에 대한 로그 손실을 의미합니다. 즉 분류에 대한 손실입니다. 회귀 손실의 경우 Lreg(ti, t²i) = R(ti - t²i)을 사용합니다. 여기서 R은 Fast R-CNN에 정의된 강력한 손실 함수(α L1)입니다. p*i Lreg는 양성 앵커에 대해서만 회귀 손실이 활성화되고(p*i = 1) 그렇지 않으면 비활성화된다는 것을 의미합니다(p*i= 0).cls 와 reg layers들은 각각 {pi}, {ti}로 출력됩니다.


# Sharing Convolutional Features for Region Proposal and Object Detection
- ImageNet 데이터로 미리 학습된 CNN M0를 준비합니다.

- M0 conv feature map을 기반으로 RPN M1를 학습합니다.
- RPN M1을 사용하여 이미지들로부터 region proposal P1을 추출합니다.
- 추출된 region proposal P1을 사용해 M0를 기반으로 Fast R-CNN을 학습하여 모델 M2를 얻습니다.
- Fast R-CNN 모델 M2의 conv feature를 모두 고정시킨 상태에서 RPN을 학습해 RPN 모델 M3을 얻습니다.
- RPN 모델 M3을 사용하여 이미지들로부터 region proposal P2을 추출합니다.
- RPN 모델 M3의 conv feature를 고정시킨 상태에서 Fast R-CNN 모델 M4를 학습합니다.
<p align='center'><img src="https://bloglunit.files.wordpress.com/2017/06/20170525-research-seminar-google-slides-2017-06-01-19-53-03.png" width="40%" height="75%" alt="process"></p>
- 앞서 말씀드린 Faster R-CNN 논문에는 기술되어 있지 않지만, 위 Alternating optimization의 복잡한 학습방법 대신, RPN의 loss function과 Fast R-CNN의 loss function을 모두 합쳐 multi-task loss로 둔 뒤, 한 번에 학습을 진행해도 Alternating optimization 방법과 거의 동일하거나 높은 성능이 나올 수 있음을 실험적으로 증명하였습니다. 이를 통해 단 한번의 학습과정으로 더욱 빠르게 Faster R-CNN 구조를 학습할 수 있습니다.

# 참조
- [https://blog.lunit.io/2017/06/01/r-cnns-tutorial/](https://blog.lunit.io/2017/06/01/r-cnns-tutorial/)
