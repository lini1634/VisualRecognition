```
import cv2

def correspondence_problem(factor):
    img1=cv2.imread("이미지1 주소",cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread("이미지2 주소",cv2.IMREAD_GRAYSCALE)
    
    sift=cv2.xfeatures2d.SIFT_create()

    # SIFT 검출 
    kp1=sift.detect(img1,None)
    kp2=sift.detect(img2,None)

    # SIFT 기술 
    kp1,des1 = sift.compute(img1,kp1) 
    kp2,des2 = sift.compute(img2,kp2)

    # FLANN 매칭
    FlANN_INDEX_KDTREE=0
    index_params=dict(algorithm=FlANN_INDEX_KDTREE,trees=5)
    search_params=dict(checks=50)

    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(des1,des2,k=2) 

    res = None
    good=[]
    for m,n in matches:
        if m.distance<factor*n.distance: 
            good.append(m)
    res=cv2.drawMatches(img1,kp1,img2,kp2,good,res,flags=2)
    
    # 이미지 출력
    img1_2,img2_2=None,None
    img1_2=cv2.drawKeypoints(img1,kp1,img1_2)
    img2_2=cv2.drawKeypoints(img2,kp2,img2_2,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('SIFT1 detect',img1_2)
    cv2.imshow('SIFT2 detect',img2_2)
    cv2.imshow('Feature Matching',res)
    cv2.waitKey(0)
    cv2.destroyAllwindows()
    
correspondence_problem(0.7)
```    


---------------------------------------------------------------------------------------------------

__corresponding problem: Detection -> Description -> Matching__

1.	Detection ->특징점 위치 파악
  
    SIFT keypoint: 위치+회전+스케일 (DOG 사용->계산 효율)
    
2.	Description ->특징점 주변 정보추출

    SIFT descriptor: Gradient 방향 Histogram -> 가우시안 ->feature vector 추출  
        + 장점: scale, 회전, 광도 변환에 불변한 descriptor  
        + 단점: 투영에 대한 이론적 대처방안x  
   
3.	Matching ->대응점 찾기  

    FLANN: Fast Library of Approximate Nearest Neighbors  
        + kd트리: 우선순위 큐와 백트래킹을 이용하여 거리가 가까운 노드부터 탐색

----------------------------------------------------------------------------------------------------

pip install opencv-python 설치  
pip install opencv-contrib-python 설치   

sift=cv2.xfeatures2d.SIFT_create()  
+ sift에 대한 함수 제공   

kp1=sift.detect(img1,None)   
+ sift의 detect를 사용하여 img1으로부터 kp1반환   

kp1,des1 = sift.compute(img1,kp1)    
+ sift의 compute를 사용하여 img1, kp1으로부터 descriptor를 계산하여 kp1과 des1 반환      
(detectAndCompute(grayimg): grayimg에서 keypoint와 descriptor 한번에 계산하고 리턴)  

FLANN 매칭을 위해 필요한 인자  
+ index_params=dict(algorithm=FlANN_INDEX_KDTREE,trees=5)  
+ search_params=dict(checks=50): feature matching을 위한 반복 횟수. checks 값이 커지면 정확한 결과값이 나오지만 속도가 느려진다     

matches=flann.knnMatch(des1,des2,k=2) #k=2 2번째로 가까운 결과까지 매칭    
factor: matches의 각 멤버에서 1순위 매칭결과가 k순위 매칭결과의 factor로 주어진 비율보다 더 가까운 값만을 취한다.  

cv2.drawKeyPoints()  
+ function which draws the small circles on the locations of keypoints.  
  
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  
+ it will draw a circle with size of keypoint and it will even show its orientation.


