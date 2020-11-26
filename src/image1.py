#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    
    #initialise a publisher to send the positions of all the joints to a topic named camera1_positions
    self.joint_positions = rospy.Publisher("/camera1/robot/joint_pos",Float64MultiArray, queue_size = 10)   
    
    #publish sin signals to Joints 2, 3 and 4 
    self.robot_joint2 = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)
    self.robot_joint3 = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size = 10)    
    self.robot_joint4 = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size = 10)    
    
     
  ####################################################################
  # Author @JamesRyan
  #
  # Description: Detect the red circle (End Effector) on the robot
  # Use a HSV colour range instead of strict RGB
  ####################################################################

  def detect_red(self,image):

    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])

    mask = cv2.inRange(hsv_img,lower_red,upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask = mask)

    M = cv2.moments(mask)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # return the masked image in colour to check against the original still images at testing
    #show_image(res)

    return np.array([cx,cy])

  ####################################################################
  # Author @JamesRyan
  #
  # Description: Detect the green circle (Joint 4) on the robot
  # Use a HSV colour range instead of strict RGB
  ####################################################################

  def detect_green(self,image):

    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_green = np.array([36,100,100])
    upper_green = np.array([86,255,255])

    mask = cv2.inRange(hsv_img,lower_green,upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask = mask)

    M = cv2.moments(mask)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # return the masked image in colour to check against the original still images at testing
    #show_image(res)

    return np.array([cx,cy])

  ####################################################################
  # Author @JamesRyan
  #
  # Description: Detect the blue circle (Joints 2 & 3) on the robot
  # Use a HSV colour range instead of strict RGB
  ####################################################################

  def detect_blue(self,image):

    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv_img,lower_blue,upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask = mask)

    M = cv2.moments(mask)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # return the masked image in colour to check against the original still images at testing
    #show_image(res)

    return np.array([cx,cy])

  ####################################################################
  # Author @JamesRyan
  #
  # Description: Detect the yellow circle (Joint 1) on the robot
  # Use a HSV colour range instead of strict RGB
  ####################################################################

  def detect_yellow(self,image):

    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])

    mask = cv2.inRange(hsv_img,lower_yellow,upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask = mask)

    M = cv2.moments(mask)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # return the masked image in colour to check against the original still images at testing
    #show_image(res)

    return np.array([cx,cy])

  ####################################################################
  # Author @JamesRyan
  #
  # Description: Detect the orange target sphere. Firstly we detect the
  # orange threshold in the image and then using the mask find the circle 
  # from the circle and rectangle in the mask
  # Use a HSV colour range instead of strict RGB
  ####################################################################

  def detect_orange_sphere(self,image):

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([1, 60, 60])
    upper_orange = np.array([18, 255, 255])

    mask = cv2.inRange(hsv_img, lower_orange, upper_orange)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    blurred_gray = cv2.blur(mask, (3, 3))
      
    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30,minRadius=1,maxRadius=100)

    
    if circles is not None:

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)
 
    #print("Center is cx :{}, cy : {}".format(center[0],center[1]))
    return np.array([center[0],center[1]])

  ####################################################################
  # Author @JamesRyan
  #
  # Description: Get the positions of all the joints, end-effector 
  # and target sphere
  ####################################################################

  def get_positions(self,image):

    pos_yellow = self.detect_yellow(image)
    pos_blue = self.detect_blue(image)
    pos_green = self.detect_green(image)
    pos_red = self.detect_red(image)
    #pos_target = self.detect_orange_sphere(image)   
    
    return np.concatenate((pos_yellow,pos_blue,pos_green,pos_red),axis=0)


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
      
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('Image1_Copy.png',self.cv_image1)

    rate = rospy.Rate(30)
    t0 = rospy.get_time()
    cur_time = 0
    
    while cur_time <5000:
      cur_time = np.array([rospy.get_time()])-t0
      auto_2 = 1.5708*np.sin(np.pi/15*cur_time/1000)
      auto_3 = 1.5708*np.sin(np.pi/18*cur_time/1000)
      auto_4 = 1.5708*np.sin(np.pi/20*cur_time/1000)
      
      joint2 = Float64()
      joint2.data = auto_2
      joint3 = Float64()
      joint3.data = auto_3
      joint4 = Float64()
      joint4.data = auto_4
      
      self.robot_joint2.publish(joint2)
      self.robot_joint3.publish(joint3)
      self.robot_joint4.publish(joint4)  
      rate.sleep() 


    #im1=cv2.imshow('Camera 1', self.cv_image1)
    #cv2.waitKey(1)
    
    joint_pos = Float64MultiArray()
    joint_pos.data = self.get_positions(self.cv_image1)
    
    
    #Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      #Publish the blob positions from Camera 1
      self.joint_positions.publish(joint_pos)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


