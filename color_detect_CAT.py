# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import csv
import imutils
import os

def write_in_csv(file,data_list):
    with open(file,'a') as file_1:
        wr = csv.writer(file_1)
        wr.writerow(data_list)

    file_1.close()


def open_csv(filio):
    with open(filio,'w') as file_1:
        pass
    
    file_1.close()


def get_good_contours(c,new_width,new_height):
    l=[]
    for j,elt in enumerate(c):
        rightmost = tuple(elt[elt[:,:,0].argmax()][0])
        topmost = tuple(elt[elt[:, :, 1].argmin()][0])
        if((rightmost[0]>3*new_width/10 and rightmost[0]<7*new_width/10) and (topmost[1]>new_height/5 and topmost[1]<1*new_height/2)):
            l.append(elt)
    return l
 
PINK_MIN = np.array([160, 80, 20],np.uint8)
PINK_MAX = np.array([179, 200, 235],np.uint8)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

open_csv(sys.argv[1])
items = os.listdir("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/CAT_again")
items.sort()
if('.DS_Store' in items):
    items.remove('.DS_Store')

for i,elt in enumerate(items,start=1):
    data_list=[]
    img=cv2.imread("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/CAT_again/"+str(elt))
    new_height, new_width, channels = img.shape 
    new_width=900
    new_height=600
    img = imutils.resize(img, height = new_height,width = new_width)

    #Detecting the moving part of the image
    # fgmask = fgbg.apply(img)
    # output = cv2.bitwise_and(img, img, mask = fgmask)
    output = img
    
    #Apply the pink filter
    output=cv2.cvtColor(output,cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(output, PINK_MIN, PINK_MAX)
    output=cv2.cvtColor(output,cv2.COLOR_HSV2BGR)

    #Display the threshed frame and the original image
    cv2.imshow('frame',frame_threshed)
    cv2.imshow('image', img)

    im2, contours, hierarchy = cv2.findContours(frame_threshed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)==0):
        stro="No contour detected for image"
        data_list.append("img_"+str(i))
        data_list.append(-1)
        data_list.append(-1)
        data_list.append(stro)
        write_in_csv(sys.argv[1],data_list)
        continue
    #Get good contours    
    contours=get_good_contours(contours,new_width,new_height)
    if(len(contours)==0):
        stro="No Good contour detected for image"
        data_list.append("img_"+str(i))
        data_list.append(-1)
        data_list.append(-1)
        data_list.append(stro)
        write_in_csv(sys.argv[1],data_list)
        continue

    #Detecting max contour & making sure it is bigger than a threshold
    c = max(contours, key = cv2.contourArea)
    if cv2.contourArea(c) < 5:
        stro="Detected contour too small"
        data_list.append("img_"+str(i))
        data_list.append(-1)
        data_list.append(-1)
        data_list.append(stro)
        write_in_csv(sys.argv[1],data_list)       
        continue

    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    #Drawing contours over the original image. Just for validation
    #cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    cX=int(x+(w/2))
    cY=int(y+(h/2))
    data_list.append("img_"+str(i))
    data_list.append(cX)
    data_list.append(cY)
    write_in_csv(sys.argv[1],data_list)

    cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the images
    cv2.imshow("Marking_centroid", img)
    cv2.imwrite('/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/CAT_again_data/output_'+str(i)+'.jpg', img)
    cv2.waitKey(0)
    