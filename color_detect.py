# import the necessary packages
import numpy as np
import argparse
import cv2
 
PINK_MIN = np.array([165, 50, 50],np.uint8)
PINK_MAX = np.array([175, 255, 255],np.uint8)

for i in range(1,33):
    #img=cv2.imread("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/Trajectory1/PG1/img_"+str(i)+".jpeg")
    img=cv2.imread("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/Trajectory1/real-sense1/img_"+str(i)+".jpeg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img = cv2.medianBlur(img,7)
    frame_threshed = cv2.inRange(img, PINK_MIN, PINK_MAX)
    #cv2.imwrite('output2.jpg', frame_threshed)
    img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    output = cv2.bitwise_and(img, img, mask = frame_threshed)
    #cv2.imshow("images", np.hstack([img, output]))

    kernel=np.ones((5,5),np.uint8)
    opening=cv2.morphologyEx(frame_threshed,cv2.MORPH_OPEN,kernel)
    cv2.imshow("opening", opening)

    #convert the image to grayscale
    gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("images", gray_image)
    # cv2.waitKey(0)
 
    # convert the grayscale image to binary image
    '''ret,thresh = cv2.threshold(gray_image,80,255,0)
    #print("Size of the binary image is ",thresh.shape)
    cv2.imshow("images", thresh)
    cv2.waitKey(0)'''
 
    # find contours in the binary image
    im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)==0):
        continue

    c = max(contours, key = cv2.contourArea)


    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imshow("Result", img)

    M = cv2.moments(c)
 
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the images
    cv2.imshow("Marking_centroid", img)
    cv2.imwrite('output_'+str(i)+'.jpg', img)
    cv2.waitKey(0)


'''
ratio = image.shape[0] / 300.0
orig = image.copy()
image = imutils.resize(image, height = 300)
'''