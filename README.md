# YOLO(You Only Look Once)!
[발표자료 다운로드(pdf)](https://github.com/dss-14th/deeplearning-repo-2/files/5764532/YOLO.You.only.look.once.pdf)

### 1. Image detection with Yolov5 

#### *✏ Before start*
- **YOLO v5 git clone**
  - https://github.com/ultralytics/yolov5
  - !git clone https://github.com/ultralytics/yolov5.git

- **package installation (requirements.txt)**
  - %cd /contents/yolov5/
  - !pip install -r requirements.txt 
  

#### *⚙ Preprocessing & training*

<img src = "https://user-images.githubusercontent.com/67793544/103855312-c3f8bc80-50f5-11eb-9769-af65e155ff59.png">

##### (1) CCTV dataset
---
- **dataset 구성**
  - train: 9 internal CCTV images of the mall
  - validation: 9 internal CCTV images in different places from train data (some overlap)
  
  ![image](https://user-images.githubusercontent.com/67793544/103861122-3b334e00-5100-11eb-86af-e4dd7aaacce2.png)


- **augmentation**
  - augment each of the train images by 11 to create 99 train images.
  - validation images are not augmented.
  
  ```
  # augmentation filter
  seq1 = iaa.Affine(scale={'x':(0.5, 1.5), 'y':(0.5, 1.5)}) # 늘리기
  seq2 = iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y':(-0.2, 0.2)}) # 옆으로 밀기
  seq3 = iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}) # 위아래로 늘리기
  seq4 = iaa.Affine(rotate=(-45, 45)) # 사진 45도 돌리기
  seq5 = iaa.Affine(shear=(-16, 16)) # 대각선으로 늘리기
  seq7= iaa.Sequential([
                      iaa.Multiply((1.2, 1.5)), 
                      iaa.Fliplr(1.0) 
                      ]) # 밝기 변화 + 좌우반전
  seq8 = iaa.Grayscale(alpha=1.0) # 회색
  seq9 = iaa.Sequential([iaa.Dropout((0.05, 0.1), per_channel=0.5),
                      iaa.Multiply((0.3, 1.5)),
                      iaa.ChannelShuffle(p=1.0)]) # dropout, 픽셀 조정
  seq10 = iaa.GaussianBlur(sigma=1.5) # 흐리게
  seq11 = iaa.Rot90(1) # 90도 회전
  ```
  
- **test 결과**

<p align='center'><img src = 'https://user-images.githubusercontent.com/67793544/103859831-163ddb80-50fe-11eb-9481-9d527986275a.gif'></p>

##### (2) BLACKPINK dataset
---
- **dataset 구성**
  - train video: 
    - (BLACKPINK Full Cam) lovesick girls full cam 
    https://www.youtube.com/watch?v=Iq6wkVsaCq0
    - BLACKPINK - ‘Lovesick Girls’ 1025 SBS Inkigayo : NO.1 OF THE WEEK
    https://www.youtube.com/watch?v=MBStYsiE618
  - test video:
    - (BLACKPINK Encore Fancam) | @SBS Inkigayo_2020.10.25.
    https://www.youtube.com/watch?v=qMsoWTlBCWc
    
- **labeling**
  - using YOLO mark
- **test 결과**

<p align='center'><img src='https://user-images.githubusercontent.com/67793544/103860464-feb32280-50fe-11eb-9499-2cd7ca2ba086.gif'> </p>

---
### 2. 참고 논문 리뷰
#### (1) [Faster R-CNN](https://github.com/dss-14th/deeplearning-repo-2/blob/main/Faster_R_Cnn_review.md)
#### (2) [YOLO v1](https://github.com/dss-14th/deeplearning-repo-2/blob/main/YOLO_v1.md)
#### (3) [YOLO v2]()
#### (4) [YOLO v3](https://github.com/dss-14th/deeplearning-repo-2/blob/main/YOLO_v3.md)
#### (5) [YOLO v3 Pytorch](https://github.com/dss-14th/deeplearning-repo-2/blob/main/YOLO_v3_torch.md)
  - YOLOv3 를 Pytorch 로 구현하는 코드는 이 [Github Repo](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)의 코드를 참고했습니다.


