#!/usr/bin/python
# -*- coding: utf-8 -*-

from E160_state import *
from E160_robot import *
import math
import time


class controller:

    def __init__(self, robot, logging=False):
        self.robot = robot  # do not delete this line
        self.kp = 0  # k_rho
        self.ka = 0  # k_alpha
        self.kb = 0  # k_beta
        self.logging = logging

        if (logging == True):
            self.robot.make_headers(['pos_X', 'posY', 'posZ', 'vix', 'viy', 'wi', 'vr', 'wr'])

        self.set_goal_points()

        # select the controller type
        self.controller_type = 'p'

    # self.controller_type='p'

    def set_goal_points(self):
        '''
        #Edit goal point list below, if you click a point using mouse, the points programmed
        #will be washed out
        '''
        # here is the example of destination code

        ##
        ##
        ##
        ## DELETE FROM THIS FILE AND PUT ONTO OUR CONTROLLER
        self.robot.state_des.add_destination(x=100, y=20, theta=0)  # goal point 1
        self.robot.state_des.add_destination(x=190, y=30, theta=1.57)  # goal point 2
        ##
        ##
        ##

    def get_robot_state(self):
        (c_posX, c_posY, c_theta) = self.robot.state.get_pos_state()  # get current position configuration
        (c_vix, c_viy,
         c_wi) = self.robot.state.get_global_vel_state()  # get current velocity configuration, in the global frame
        (c_v, c_w) = self.robot.state.get_local_vel_state()  # get current local velocity configuration
        return c_posX, c_posY, c_theta, c_vix, c_viy, c_wi, c_v, c_w

    def track_point(self):
        '''
        Main controller method for tracking point
        '''

        ## The following demonstrate how to get the state of the robot

        # All d_ means destination
        (d_posX, d_posY, d_theta) = self.robot.state_des.get_des_state()  # get next destination configuration

        # All c_ means current_
        c_posX, c_posY, c_theta, c_vix, c_viy, c_wi, c_v, c_w = self.get_robot_state()  # get current robot state

        ## Most of your program should be here, compute rho, alpha and beta using d_pos and c_pos
        if (self.controller_type == 'a'):
            distX = d_posX - c_posX
            distY = d_posY - c_posY
            alpha = math.atan2(distY, distX)
            distTheta = alpha - c_theta
            distRho = math.hypot(distX, distY)
            distTheta = math.atan2(math.sin(distTheta), math.cos(distTheta))  # Normalize to -pi to pi
            goalAngle = d_theta - c_theta
            goalAngle = math.atan2(math.sin(goalAngle), math.cos(goalAngle))  # Normalize to -pi to pi
            # if it is "a controller", ensure the wheel speed is not violated, you need to
            # 1. turn your robot to face the target position
            if abs(distTheta) > 0.05:  # Within 0.05 rad, about 3 degree
                self.robot.set_motor_control(0, 0.5)
            # 2. move your robot forward to the target position
            elif abs(distRho) > 0.1:  # Within 0.1 cm
                self.robot.set_motor_control(10, 0)
            # 3. turn your robot to face the target orientation
            elif abs(goalAngle) > 0.05:  # Within 0.05 rad, about 3 degree
                self.robot.set_motor_control(0, 0.5)
            else:
                self.robot.set_motor_control(0, 0)
                print("Goal reached")

        elif (self.controller_type == 'p'):
            # use p control discussed in the class,
            # set new c_v = k_rho*rho, c_w = k_alpha*alpha + k_beta*beta

            distX = d_posX - c_posX
            distY = d_posY - c_posY
            distRho = math.hypot(distX, distY)  # shortest distance to goal

            alpha = math.atan2(distY, distX) #this is the angle of the robot off the x axis
            alphaError = alpha - c_theta  #difference between the angle of the robot to the goal and the robots current orientation
            alphaError = math.atan2(math.sin(alphaError), math.cos(alphaError))#this wraps the direction of the robot, so it always turns the shortest way


            betaError = d_theta - alpha
            betaError = math.atan2(math.sin(betaError), math.cos(betaError)) #this just does the wrapping similar to the a controller

            #define our gain
            self.kp = 6
            self.ka = 5
            self.kb = -0.5
            c_v = self.kp * distRho
            c_w = self.ka * alphaError + self.kb * betaError

            # define max speeds so our robot doesn't break
            v_max = 16
            w_max = 16

            if c_v > v_max:
                c_v = v_max
            elif c_v < -v_max:
                c_v = -v_max

            if c_w > w_max:
                c_w = w_max
            elif c_w < -w_max:
                c_w = -w_max





            #now actually get the robot moving
            self.robot.set_motor_control(c_v, c_w)

            if distRho < 0.2:
                self.robot.set_motor_control(0, 0)
                print("Goal Reached!")
                self.robot.set_motor_control(0, 0)
                return True
            else:
                print('Continuing to point...')



        else:
            print("no controller type is provided")
            c_v = 0
            c_w = 0

        # self.robot.set_motor_control(linear velocity (cm), angular velocity (rad))
        self.robot.set_motor_control(c_v, c_w)  # use this command to set robot's speed in local frame

        # you need to write code to find the wheel speed for your c_v, and c_w, the program won't calculate it for you.

        self.robot.send_wheel_speed(phi_l=6.0, phi_r=6.0)  # unit rad/s

        # use the following to log the variables, use [] to bracket all variables you want to store
        # stored values are in log folder
        if self.logging == True:
            self.robot.log_data([c_posX, c_posY, c_theta, c_vix, c_viy, c_wi, c_v, c_w])

        if abs(c_posX - d_posX) < 5 and abs(c_theta - d_theta) < 0.2:  # you need to modify the reach way point criteria
            if (self.robot.state_des.reach_destination()):
                print("final goal reached")
                self.robot.set_motor_control(.0, .0)  # stop the motor
                return True
            else:
                print("one goal point reached, continute to next goal point")

        return False  # just return False, don't remove it.