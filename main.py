from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import numpy as np
import argparse
import imutils
import time
import random
from sklearn.decomposition import RandomizedPCA
from itertools import cycle
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from sklearn.externals import joblib
svc = joblib.load('ML/theta.pkl')
def bgsub():
    init_()
    cam=cv2.VideoCapture(0)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    img1 = cv2.imread('pho.jpg')
    while(cam.isOpened): 
       f,img=cam.read()
       if f==True:
           
           #img=cv2.flip(img,1)
           #img=cv2.medianBlur(img,3)
           fgmask = fgbg.apply(img)
           fgmask = cv2.resize(fgmask, (360,240), interpolation = cv2.INTER_CUBIC)
           top,down,left,right = getrec(fgmask)
           img=cv2.resize(img,(1080,720),interpolation = cv2.INTER_CUBIC)
           #frm = cv2.rectangle(img,(left*3,top*3),(right*3,down*3),(255,255,0),2)
           #roi = frm[top*3:(down-1)*3,left*3:(right-1)*3]
           #roib1 =fgmask[top:down-1,left:right-1]
           #roib = cv2.resize(roib1,(60,60),interpolation = cv2.INTER_CUBIC)
           #cv2.imwrite('tempph/'+str(random.randint(1,1000000))+'.jpg',roib)
           frm = img[top*3:(down-1)*3,left*3:(right-1)*3]
           frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
           #gray = cv2.GaussianBlur(roib,(9,9),0)
           #re,gray = cv2.threshold(roib,70,255,cv2.THRESH_BINARY)
           #####background
           initimg = cv2.imread('background.jpg')
           initimgg = cv2.cvtColor(initimg, cv2.COLOR_BGR2GRAY)
           initimgg = cv2.resize(initimgg,(1080,720),interpolation = cv2.INTER_CUBIC)
     	   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           temp  = cv2.subtract(initimgg,img)
           ttt = cv2.bitwise_and(img,initimgg);
     	   ttt = cv2.bitwise_not(ttt)
     	   re,ttt = cv2.threshold(ttt,120,255,cv2.THRESH_BINARY)
           re,temp = cv2.threshold(temp,40,255,cv2.THRESH_BINARY)
           #temp = cv2.resize(temp,(60,60),interpolation = cv2.INTER_CUBIC)
          # cv2.fastNlMeansDenoising(temp,temp,None,10,7,21)
           temp = cv2.GaussianBlur(temp,(9,9),0)
           re,temp = cv2.threshold(temp,120,255,cv2.THRESH_BINARY)
           #temp = temp[0:29,0:59]
           temp = cv2.resize(temp,(120,120),interpolation = cv2.INTER_CUBIC)
           temp = cv2.GaussianBlur(temp,(9,9),0)
           re,temp = cv2.threshold(temp,80,255,cv2.THRESH_BINARY)
           #t,d,l,r = getrec(temp)
           temp = temp[0:59,0:119]
           t,d,l,r = getrec(temp)
           temp = temp[t:d-1,l:r-1]
           temp = cv2.resize(temp, (120,60), interpolation = cv2.INTER_CUBIC)
           temp = cv2.GaussianBlur(temp,(9,9),0)
           re,temp = cv2.threshold(temp,80,255,cv2.THRESH_BINARY)
           #if(t and d and l and r):
          	 #temp = temp[0:59,left:right-1]
           testx = destruct(temp)
           print(svc.predict(testx))
           if(svc.predict(testx)==1):
                #cv2.imwrite('success/'+str(random.randint(1,1000000))+'.jpg',img)
                print("success")
           cv2.imwrite('tempph/'+str(random.randint(1,1000000))+'.png',temp)

           #time.sleep(0.3)
           #gray = cv2.bitwise_not(gray)
           #f = frm[top,(down-1),left:(right-1)]
           #gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
           cv2.imshow('track',temp)
           #cv2.imshow('track1',gray)
           #cv2.imshow('roi',initimgg)
           #cv2.imshow('frame',img)
       if(cv2.waitKey(27)!=-1):
           cam.release()
           cv2.destroyAllWindows()
           #break
def main():
    bgsub()
def getrec(fgmask):
    top,down,left,right =0,0,0,0
    shape = fgmask.shape
    shape = (shape[1],shape[0])
    bol = False
    k=shape[1]
    for i in range(shape[1]):
        for j in range(shape[0]):
            if(fgmask[i][j]==255 ):
                top = i
                bol = True
                break
        if(bol):
            break
    bol = False
    for i in range(shape[1]):
        for j in range(shape[0]):
            if(fgmask[k-i-1][j]==255):
                down = k-i-1
                bol = True
                break
        if(bol):
            break
    bol = False
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(fgmask[j][i]==255):
                left = i
                bol = True
                break
        if(bol):
            break
    bol = False
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(fgmask[j][shape[0]-i-1]==255):
                right = shape[0]-i-1
                bol = True
                break
        if(bol):
            break
    return top,down,left,right
def init_():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    ret, frame = cap.read()
    res = cv2.resize(frame,(40,40),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('pho.jpg',res)
    cv2.imwrite('background.jpg',frame)
       
    cap.release()
    cv2.destroyAllWindows()
def destruct(img):
    arr = np.array(img)
    flat = arr.ravel()


    return flat
main()



