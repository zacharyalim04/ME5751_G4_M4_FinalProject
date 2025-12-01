#!/usr/bin/python
# -*- coding: utf-8 -*-

from E160_state import *
from E160_robot import *
import math
import time


class controller:

    def __init__(self, robot, logging=False):
        self.graphics = None

        self.robot = robot  # do not delete this line
        self.kp = 6   # k_rho
        self.ka = 3   # k_alpha
        self.kb = -0.3  # k_beta
        self.logging = logging

        if logging:
            self.robot.make_headers(['pos_X', 'posY', 'posZ', 'vix', 'viy', 'wi', 'vr', 'wr'])

        # select the controller type
        self.controller_type = 'a'

    def attach_graphics(self, graphics):
        self.graphics = graphics   
        self.set_goal_points() 

    # --------------------------------------------------
    # Load PRM path as goals for the controller
    # --------------------------------------------------
    def set_goal_points(self):
        """
        Load PRM path as goals for the controller.
        """
        import csv

        # Clear existing goals in E160_des_state
        self.robot.state_des.x = []
        self.robot.state_des.y = []
        self.robot.state_des.theta = []
        self.robot.state_des.p = 0

        print("Loading PRM path...")

        with open("Log/prm_path.csv", newline='') as f:
            reader = csv.reader(f, delimiter="\t")

            next(reader)  # skip header

            for row in reader:
                if len(row) < 3:
                    continue

                map_i = int(float(row[0]))
                map_j = int(float(row[1]))
                theta = float(row[2])

                # Convert using PRM’s map2world
                world_x, world_y = self.graphics.path.map2world(map_i, map_j)

                self.robot.state_des.add_destination(x=world_x, y=world_y, theta=theta)

        print("PRM path loaded successfully.")

    # --------------------------------------------------
    def get_robot_state(self):
        (c_posX, c_posY, c_theta) = self.robot.state.get_pos_state()
        (c_vix, c_viy, c_wi) = self.robot.state.get_global_vel_state()
        (c_v, c_w) = self.robot.state.get_local_vel_state()
        return c_posX, c_posY, c_theta, c_vix, c_viy, c_wi, c_v, c_w

    # --------------------------------------------------
    def track_point(self):
        """
        Main controller method for tracking point
        """

        # Debug if you want, but not every step:
        # print("DEST INDEX:", self.robot.state_des.p)
        # print("NUM DEST:", len(self.robot.state_des.x))

        # If no destinations loaded yet, don't run controller
        if len(self.robot.state_des.x) == 0:
            return False

        # All waypoints already consumed
        if self.robot.state_des.p >= len(self.robot.state_des.x):
            # print("All waypoints completed; nothing left to track.")
            self.robot.set_motor_control(0, 0)
            return True

        # Desired state
        (d_posX, d_posY, d_theta) = self.robot.state_des.get_des_state()

        # Current state
        c_posX, c_posY, c_theta, c_vix, c_viy, c_wi, c_v, c_w = self.get_robot_state()


        # --------------------------------------------------
        # LOOKAHEAD-SELECTION: Pure Pursuit style tracking
        # --------------------------------------------------

        lookahead_dist = 30.0   # tune 60–120 px

        # Find index of the closest waypoint to the robot
        closest_idx = 0
        best_dist = float('inf')
        for i in range(len(self.robot.state_des.x)):
            wx = self.robot.state_des.x[i]
            wy = self.robot.state_des.y[i]
            d = math.hypot(wx - c_posX, wy - c_posY)

            if d < best_dist:
                best_dist = d
                closest_idx = i

        # Now walk FORWARD along the path until we accumulate lookahead_dist
        look_idx = closest_idx
        accum = 0.0

        for i in range(closest_idx, len(self.robot.state_des.x)-1):
            x1, y1 = self.robot.state_des.x[i], self.robot.state_des.y[i]
            x2, y2 = self.robot.state_des.x[i+1], self.robot.state_des.y[i+1]

            seg = math.hypot(x2 - x1, y2 - y1)
            accum += seg

            if accum >= lookahead_dist:
                look_idx = i + 1
                break

        # Clamp to final waypoint if we run out of path
        look_idx = min(look_idx, len(self.robot.state_des.x)-1)

        # This is the waypoint the robot should track
        d_posX = self.robot.state_des.x[look_idx]
        d_posY = self.robot.state_des.y[look_idx]
        d_theta = self.robot.state_des.theta[look_idx]


        # --------------------------------------------------
        # “a” controller (unused but left intact)
        # --------------------------------------------------
        if self.controller_type == 'a':

            # --- Position errors ---
            distX = d_posX - c_posX
            distY = d_posY - c_posY
            alpha = math.atan2(distY, distX)

            distTheta = alpha - c_theta
            distTheta = math.atan2(math.sin(distTheta), math.cos(distTheta))
            distRho = math.hypot(distX, distY)

            goalAngle = d_theta - c_theta
            goalAngle = math.atan2(math.sin(goalAngle), math.cos(goalAngle))

            # -------------------------------------------------
            # FIX 4: "Stop Box" near final goal
            # -------------------------------------------------
            final_x = self.robot.state_des.x[-1]
            final_y = self.robot.state_des.y[-1]
            final_theta = self.robot.state_des.theta[-1]

            final_dist = math.hypot(final_x - c_posX, final_y - c_posY)
            final_angle_err = math.atan2(math.sin(final_theta - c_theta),
                                        math.cos(final_theta - c_theta))

            # If very close to final point: freeze lookahead and refine
            if final_dist < 20: # TUNE FOR GOAL DISTANCE APPLICATION
                d_posX = final_x
                d_posY = final_y
                d_theta = final_theta

                goalAngle = math.atan2(math.sin(d_theta - c_theta),
                           math.cos(d_theta - c_theta))

                # If extremely close → fully stop
                if final_dist < 8 and abs(final_angle_err) < 0.10:
                    self.robot.set_motor_control(0, 0)
                    print("Goal reached cleanly")
                    self.robot.state_des.p = len(self.robot.state_des.x)
                    return True

            # --- BASIC PROPORTIONAL CONTROLLER ---
            # angular velocity
            c_w = 3.0 * distTheta * min(1.0, distRho / 40.0)       # ka = 3
            # linear speed
            c_v = 35.0 * math.tanh(distRho / 25.0)

            # --- SLOW DOWN NEAR FINAL GOAL ---
            goal_dist = math.hypot(self.robot.state_des.x[-1] - c_posX,
                                self.robot.state_des.y[-1] - c_posY)

            if goal_dist < 30:
                c_v = max(8.0, 0.3 * goal_dist)

            # -------------------------------------------
            # FIX 3: Stronger final-orientation control
            # -------------------------------------------
            # When close to the final waypoint, blend in heading correction
            if goal_dist < 30: # TUNE THE GOAL DISTANCE APPLICATION
                w_goal = 2.0 * goalAngle      # try 1.5–3.0
                blend = max(0.0, 1.0 - goal_dist / 80.0)   # fades in smoothly
                c_w = (1 - blend) * c_w + blend * w_goal


            # -------------------------------
            # ACKERMANN STEERING LIMITS
            # -------------------------------
            if self.robot.state.vehicle == "v":
                d = self.robot.state.d
                beta_max = self.robot.state.beta_max

                if abs(c_v) < 1e-3:
                    beta = 0
                else:
                    beta = math.atan(c_w * d / c_v)

                if abs(beta) > beta_max:
                    beta = math.copysign(beta_max, beta)
                    c_w = math.tan(beta) * c_v / d

            # send the valid control
            self.robot.set_motor_control(c_v, c_w)

            return False


        # --------------------------------------------------
        # P controller that tracks PRM waypoints
        # --------------------------------------------------
        elif self.controller_type == 'p':

            distX = d_posX - c_posX
            distY = d_posY - c_posY
            distRho = math.hypot(distX, distY)  # distance to goal

            alpha = math.atan2(distY, distX)
            alphaError = alpha - c_theta
            alphaError = math.atan2(math.sin(alphaError), math.cos(alphaError))

            betaError = d_theta - alpha
            betaError = math.atan2(math.sin(betaError), math.cos(betaError))

            # basic P gains
            # --------------------------------------------------
            # NEW SPEED LOGIC: cruise fast between waypoints,
            # slow only near the final destination
            # --------------------------------------------------

            # Distance to *final* goal (not the next waypoint)
            goal_dist = math.hypot(
                self.robot.state_des.x[-1] - c_posX,
                self.robot.state_des.y[-1] - c_posY
            )

            # If close to final goal, stop using lookahead (prevents weaving)
            if goal_dist < 40:    # tune 25–50
                d_posX = self.robot.state_des.x[-1]
                d_posY = self.robot.state_des.y[-1]
                d_theta = self.robot.state_des.theta[-1]


            # Are we near the final goal?
            if goal_dist < 60:        # slow-down radius (tune 40–80)
                # Smooth deceleration only at the end
                c_v = max(10.0, 0.5 * goal_dist)
            else:
                # Cruise at full speed between waypoints
                c_v = 30.0

            c_w = self.ka * alphaError + self.kb * betaError

            # --- linear/angular speed limits from project spec ---
            v_max = 40.0
            w_max = 16.0

            if c_v > v_max:
                c_v = v_max
            elif c_v < -v_max:
                c_v = -v_max

            if c_w > w_max:
                c_w = w_max
            elif c_w < -w_max:
                c_w = -w_max

            # --------------------------------------------------
            # *** Ackermann steering constraint ***
            # limit curvature w/v so that |beta| <= beta_max
            # --------------------------------------------------
            if self.robot.state.vehicle == "v":
                d = self.robot.state.d
                beta_max = self.robot.state.beta_max

                # --- Ackermann steering conversion ---
                if abs(c_v) < 1e-3:
                    beta = 0
                else:
                    beta = math.atan(c_w * d / c_v)

                # enforce steering limits
                if abs(beta) > beta_max:
                    beta = math.copysign(beta_max, beta)
                    # recompute allowed angular velocity
                    c_w = math.tan(beta) * c_v / d

            # send command to robot
            self.robot.set_motor_control(c_v, c_w)

            # Waypoint reached?
            # (tolerance in world coords – tune if needed)
            if distRho < 10.0:
                # print("Reached waypoint.")
                if self.robot.state_des.reach_destination():
                    print("Final goal reached.")
                    self.robot.set_motor_control(0, 0)
                    return True
                else:
                    # print("Advancing to next waypoint...")
                    return False

        else:
            print("No valid controller type selected — stopping robot.")
            self.robot.set_motor_control(0, 0)

        # Logging
        if self.logging:
            self.robot.log_data([c_posX, c_posY, c_theta, c_vix, c_viy, c_wi, c_v, c_w])

        return False