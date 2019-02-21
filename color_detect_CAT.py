# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import csv
import imutils
import os

'''
System argument requirements:
1. The CSV file on which the rover's centroid coordinates would be published (Provide the absolute path. Else it will generate the file in the working directory)
'''

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
    '''
    INPUT: Takes in the contour and the height and width of the image
    OUTPUT: Returns the contours which lie within the specified screen range provided
    '''
    l=[]
    for j,elt in enumerate(c):
        rightmost = tuple(elt[elt[:,:,0].argmax()][0])
        topmost = tuple(elt[elt[:, :, 1].argmin()][0])
        if((topmost[1]>3*new_height/10 and topmost[1]<7*new_height/10) and (rightmost[0]>(0.4*new_width))):
            l.append(elt)
    return l
 
PINK_MIN = np.array([160, 80, 20],np.uint8)
PINK_MAX = np.array([179, 200, 235],np.uint8)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

open_csv(sys.argv[1])
#CHANGE THIS PATH AS PER THE USE
items = os.listdir("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/New_CATA_field_exp/Dataset4/remove_top_up")
items.sort()
if('.DS_Store' in items):
    items.remove('.DS_Store')

for i,elt in enumerate(items,start=1):
    '''
    INPUT: Takes in all the images in the requested folder and applies detection algorithm on the same.
    USE: Saves the centroid marked images in the folder specified by the user
    OUTPUT: -
    '''
    data_list=[]
    img=cv2.imread("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/New_CATA_field_exp/Dataset4/remove_top_up/"+str(elt))     #CHANGE THIS PATH AS PER THE USE
    new_height, new_width, channels = img.shape 
    new_height=600
    new_width=900
    img = imutils.resize(img, height = new_height,width = new_width)

    #Detecting the moving part of the image
    # fgmask = fgbg.apply(img)
    # output = cv2.bitwise_and(img, img, mask = fgmask)
    output = img
    
    #Apply the pink filter
    output=cv2.cvtColor(output,cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(output, PINK_MIN, PINK_MAX)
    output=cv2.cvtColor(output,cv2.COLOR_HSV2BGR)

    '''
    DISPLAYING OF IMAGES AND THRESHOLD
    '''
    #Display the threshed frame and the original image
    #cv2.imshow('frame',frame_threshed)
    #cv2.imshow('image', img)

    im2, contours, hierarchy = cv2.findContours(frame_threshed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)==0):
        stro="No contour detected for image"
        data_list.append(str(elt))
        data_list.append(-1)
        data_list.append(-1)
        data_list.append(stro)
        write_in_csv(sys.argv[1],data_list)
        continue
    #Get good contours    
    contours=get_good_contours(contours,new_width,new_height)
    if(len(contours)==0):
        stro="No Good contour detected for image"
        data_list.append(str(elt))
        data_list.append(-1)
        data_list.append(-1)
        data_list.append(stro)
        write_in_csv(sys.argv[1],data_list)
        continue

    #Detecting max contour & making sure it is bigger than a threshold
    c = max(contours, key = cv2.contourArea)
    if cv2.contourArea(c) < 0:
        stro="Detected contour too small"
        data_list.append(str(elt))
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
    data_list.append(str(elt))
    data_list.append(cX)
    data_list.append(cY)
    write_in_csv(sys.argv[1],data_list)

    cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the images
    '''
    DISPLAYING OF IMAGES AND THRESHOLD
    '''
    cv2.imshow("Marking_centroid", img)

    #Below I write the centroid marked images in the output folder. Change directory as per use
    cv2.imwrite('/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/New_CATA_field_exp/Dataset4/output_remove_left/'+str(elt)+'.jpg', img)        #CHANGE THIS PATH AS PER THE USE
    cv2.waitKey(0)
    