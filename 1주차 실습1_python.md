import numpy as np
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
    
    
