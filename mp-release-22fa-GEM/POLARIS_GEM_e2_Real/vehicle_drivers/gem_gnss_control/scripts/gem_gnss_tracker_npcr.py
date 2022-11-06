#!/usr/bin/env python3

#==============================================================================
# File name          : gem_gnss_tracker_npcr.py (nick, pramod, charlotte, ralph)                                                                  
# Description        : gnss waypoints tracker using pid and Stanley controller                                                              
# Author             : Ralph Balita (rbalita2@illinois.edu)                                       
# Date created       : 11/02/2022 (post 391 exam, def failed, here doing 484 to make my day better)                                                                 
# Date last modified : TBA                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_gnss_control gem_gnss_tracker_npcr.py                                                                   
# Python version     : 3.8   
# Longitudinal ctrl  : Nick Stone (nm14), Pramod Prem(pprem2), Charlotte Fondren (fondren3)                                                            
#==============================================================================

'''
# libraries from 
# gem_gnss_tracker_pp.py 
# and gem_gnss_tracker_stanley_rtk.py 
'''
from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy
import alvinxy.alvinxy as axy 
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

'''
libraries from 
gem_gnss_tracker_pp.py 
'''
# GEM Sensor Headers - used in gem_gnss_tracker_pp.py (pure_pursuit)
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva


'''
# common controller code from 
# gem_gnss_tracker_pp.py 
# and gem_gnss_tracker_stanley_rtk.py 
'''
class PID(object):

  def __init__(self, kp, ki, kd, wg=None):

      self.iterm  = 0
      self.last_t = None
      self.last_e = 0
      self.kp     = kp
      self.ki     = ki
      self.kd     = kd
      self.wg     = wg
      self.derror = 0

  def reset(self):
      self.iterm  = 0
      self.last_e = 0
      self.last_t = None

  def get_control(self, t, e, fwd=0):

      if self.last_t is None:
          self.last_t = t
          de = 0
      else:
          de = (e - self.last_e) / (t - self.last_t)

      if abs(e - self.last_e) > 0.5:
          de = 0

      self.iterm += e * (t - self.last_t)

      # take care of integral winding-up
      if self.wg is not None:
          if self.iterm > self.wg:
              self.iterm = self.wg
          elif self.iterm < -self.wg:
              self.iterm = -self.wg

      self.last_e = e
      self.last_t = t
      self.derror = de

      return fwd + self.kp * e + self.ki * self.iterm + self.kd * de

class OnlineFilter(object):

  def __init__(self, cutoff, fs, order):
      
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    # Get the filter coefficients 
    self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Initialize
    self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
      filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
      return filted

'''
# the first controller i have ever written by myself
# mark this day: Nov 2, 2022 2:28AM
# first code ever written for an autonomous vehicle
# the start of my journey to in AV starts right now
# absolutely high out of my mind
'''

# ----- MACROS -----
RATE      = 10        # or 30
REF_SPEED = 1.5
MAX_ACCEL = 0.48

ORIG_OFFSET = 1.1 # or 0.46

HEADING_MIN = 0       # or 270
HEADING_MAX = 90      # or 360


# -----------------

'''
========== NOTES ==========
(1) read_waypoints(self) 
-> may or may not need wp_size (waypoint size) and dist_arr (arr of error dist)

(2) wps_to_local_xy_npcr(self, lon_wp, lat_wp) 
-> may or may not need to return the negative results for lon_wp_x (longitude_waypoint_x) and lat_wp_y (latitude waypoint_y)  

(3) heading_to_yaw_npcr(self, heading_curr)
-> may or may not need to change the HEADING_MIN and HEADING_MAX
-> may or may not need to change the equations for the updated yaw values 

(4) get_gem_state(self)
-> may or may not need to change the equations for curr_x & curr_y (minus or plus)
# -----------------
'''



class NPCR(object):
  def __init__(self):

    # --- refresh rate ---
    self.rate   = rospy.Rate(RATE)

    # --- init subscribers: gnss, pacmod, speed ---
    self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
    self.lat        = 0.0
    self.lon        = 0.0
    self.heading    = 0.0

    self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

    self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
    self.speed      = 0.0

    # --- offset ---
    self.offset = ORIG_OFFSET # meters

    # --- original latitude and longitude ---
    self.olat       = 40.0928563
    self.olon       = -88.2359994

    # --- PID for longitudinal control ---
    self.desired_speed = REF_SPEED                  # m/s, reference speed
    self.max_accel     = MAX_ACCEL                  # % of acceleration
    self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
    self.speed_filter  = OnlineFilter(1.2, 30, 4)

    # --- reading waypoints into the system ---
    self.read_waypoints()

  '''
  Getter functions (callback functions)
  '''
  # GNSS (from invspa_msg)
  def inspva_callback(self, inspva_msg):
    self.lat     = inspva_msg.latitude  # latitude
    self.lon     = inspva_msg.longitude # longitude
    self.heading = inspva_msg.azimuth   # heading in degrees

  # Vehicle speed
  def speed_callback(self, msg):
    self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

  # Steering wheel
  def steer_callback(self, msg):
    self.steer = round(np.degrees(msg.output),1)

  # PACMod Enable Callback
  def enable_callback(self, msg):
    self.pacmod_enable = msg.data

  # Predefined waypoints based on GNSS
  def read_waypoints(self):
    # read recorded GPS lat, lon, heading
    dirname  = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../waypoints/xyhead_demo_pp.csv')
    with open(filename) as f:
      path_points = [tuple(line) for line in csv.reader(f)]
    # x towards East and y towards North
    self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
    self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
    self.path_points_heading = [float(point[2]) for point in path_points] # heading
    self.wp_size             = len(self.path_points_lon_x)
    self.dist_arr            = np.zeros(self.wp_size)
  

  '''
  Conversion Functions
  '''
  # Conversion of front wheel to steering wheel
  def front2steer(self, f_angle):
    if(f_angle > 35):
      f_angle = 35
    if (f_angle < -35):
      f_angle = -35
    if (f_angle > 0):
      steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
    elif (f_angle < 0):
      f_angle = -f_angle
      steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
    else:
      steer_angle = 0.0
    return steer_angle
  
  # Computes the Euclidean distance between two 2D points
  def dist(self, p1, p2):
    return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

  '''
  Conversion Functions specific to NPCR
  '''
  # Conversion of Lon & Lat to X & Y
  def wps_to_local_xy_npcr(self, lon_wp, lat_wp):
    # convert GNSS waypoints into local fixed frame reprented in x and y
    lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
    return lon_wp_x, lat_wp_y   

  # Conversion of GNSS heading to vehicle heading
  def heading_to_yaw_npcr(self, heading_curr):
    if (heading_curr >= HEADING_MIN and heading_curr < HEADING_MAX):
      yaw_curr = np.radians(-heading_curr-90)
    else:
      yaw_curr = np.radians(-heading_curr+270)
    return yaw_curr
  
  
  
  '''
  Getter functions specific to NPCR
  '''
  # Get vehicle states: x, y, yaw
  def get_gem_state(self):

    # vehicle gnss heading (yaw) in degrees
    # vehicle x, y position in fixed local frame, in meters
    # rct_errorerence point is located at the center of GNSS antennas
    local_x_curr, local_y_curr = self.wps_to_local_xy_npcr(self.lon, self.lat)

    # heading to yaw (degrees to radians)
    # heading is calculated from two GNSS antennas
    curr_yaw = self.heading_to_yaw_npcr(self.heading) 

    # rct_errorerence point is located at the center of front axle
    curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
    curr_y = local_y_curr + self.offset * np.sin(curr_yaw)

    return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

  '''
  The main course: the thing I will be working on for a minute
  '''
  # Start Stanley controller
  def start_npcr(self):
    while not rospy.is_shutdown():
      # --- sanity checks ---
      if self.gem_enable != True:
        break
      
      # --- start up sequence --- 
      if self.pacmod_enable == True:

          # enable forward gear
          self.gear_cmd.ui16_cmd = 3

          # enable brake
          self.brake_cmd.enable  = True
          self.brake_cmd.clear   = False
          self.brake_cmd.ignore  = False
          self.brake_cmd.f64_cmd = 0.0

          # enable gas 
          self.accel_cmd.enable  = True
          self.accel_cmd.clear   = False
          self.accel_cmd.ignore  = False
          self.accel_cmd.f64_cmd = 0.0

          self.gear_pub.publish(self.gear_cmd)
          print("Forward Gear Engaged, CHECK")

          self.turn_pub.publish(self.turn_cmd)
          print("Turn Signal Engaged, CHECK")
          
          self.brake_pub.publish(self.brake_cmd)
          print("Brake Pedal Engaged, CHECK")

          self.accel_pub.publish(self.accel_cmd)
          print("Gas Pedal Engaged, CHECK")

          print("Ready")
          print("Set")
          print("IGNITION!")

          self.gem_enable = True
      
      '''
      DO STUFF HERE
      '''
      
      # --- publish commands to PACMod ---
      self.accel_cmd.f64_cmd = output_accel
      self.steer_cmd.angular_position = np.radians(steering_angle)
      self.accel_pub.publish(self.accel_cmd)
      self.steer_pub.publish(self.steer_cmd)
      self.turn_pub.publish(self.turn_cmd)

      self.rate.sleep()
  
  
  


'''
our main function to initialize the 
controller and call its startup function
'''
def npcr_run():

  rospy.init_node('gnss_npcr_node', anonymous=True)
  nprc = NPCR()

  try:
    nprc.start_npcr()
  except rospy.ROSInterruptException:
    pass


if __name__ == '__main__':
    npcr_run()