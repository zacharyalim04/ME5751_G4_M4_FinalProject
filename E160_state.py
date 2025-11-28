import numpy as np
import math

class E160_state:
	def __init__(self, vehicle = "d"):
		self.x = .0
		self.y = .0
		self.theta = .0
		self.vix = .0 #velocity global frame x direction
		self.viy = .0 #velocity global frame y direction
		self.wi = .0 #angular velocity, global frame
		self.v = .0 #velocity, robot frame x direction
		self.w = .0 #angular velocity, robot frame
		self.vehicle = vehicle
		self.phi_l = .0 #wheel speed, left, reported by user
		self.phi_r = .0 #wheel speed, right, reported by user

		if self.vehicle == "d":
			self.L = 12.0 #dimension of the robot
			self.r = 3.0 #dimension of the wheel
			self.phi_max = 16 #max velocity of the wheel
			self.trans = np.array([[0.5*self.r,0.5*self.r],[0.5*self.r/self.L,-0.5*self.r/self.L]])
			self.trans_inv = np.linalg.inv(self.trans)
		elif self.vehicle == "v":
			self.phi_max = 20.0 #max velocity of the wheels
			self.Rt_max = 32.0 #max radius of curvature
			self.d = 20.0 #dimension of the robot
			self.beta_max = math.asin(self.d/self.Rt_max) # max steering angle
			self.L = 8.0
			self.r = 3.0


		self.wrong_speed = False

	def set_pos_state(self,x,y,theta):
		self.x = x
		self.y = y
		self.theta = theta

	def get_pos_state(self):
		return self.x, self.y, self.theta


	def get_global_vel_state(self):
		return self.vix, self.viy, self.wi


	def get_local_vel_state(self):
		return self.v, self.w

	# set the velocity in local frame, we have code to convert it to global frame, not sure the mode for
	def set_vel_state(self, v, w, deltaT = 0.1, mode = "S"):
		self.v = v
		self.w = w
		if (mode == 'S'):
			self.update_pos_state(deltaT, vehicle = self.vehicle)


	# set the velocity in 
	def set_wheel_speed(self, phi_l, phi_r):
		self.phi_l = phi_l
		self.phi_r = phi_r


	# transfer local velocity to global velocity
	def _get_global_velocity(self,dx,dy):
		# velocity transform matrix T_v
		T_v = np.array([[math.cos(self.theta),-math.sin(self.theta),0],[math.sin(self.theta),math.cos(self.theta),0],[0,0,1]])

		#local velocity vector
		v_v = np.array([dx,dy,.0])

		dxi,dyi,_=T_v @ v_v
		return dxi,dyi


	def update_pos_state(self, deltaT, vehicle ="d"):
		self._find_wheel_speed(vehicle = vehicle) 

		ds = self.v * deltaT
		dtheta = self.w * deltaT
		dx = ds * math.cos(dtheta/2.0)
		dy = ds * math.sin(dtheta/2.0)

		dxi, dyi = self._get_global_velocity(dx,dy)

		self.x = self.x + dxi
		self.y = self.y + dyi
		self.theta = self.theta + dtheta
		if(self.theta>math.pi):
			self.theta -= math.pi*2
		elif(self.theta<-math.pi):
			self.theta += math.pi*2


	def _find_wheel_speed(self, vehicle = "d"):
		self.wrong_speed = False

		if vehicle == "d":
			vec = np.array([self.v, self.w])
			phi_r, phi_l = self.trans_inv @ vec
			# print(phi_r, phi_l)
			self.wrong_speed = False

			if(phi_r>self.phi_max):
				phi_r = self.phi_max
				self.wrong_speed = True

			elif(phi_r<-self.phi_max):
				phi_r = -self.phi_max
				self.wrong_speed = True

			if(phi_l>self.phi_max):
				phi_l = self.phi_max
				self.wrong_speed = True

			elif(phi_l<-self.phi_max):
				phi_l = -self.phi_max
				self.wrong_speed = True

			[self.v,self.w]=self.trans @ np.array([phi_r,phi_l])

		if vehicle == "v":
			if(self.v >self.phi_max*self.r):
				self.v = self.phi_max*self.r
				self.wrong_speed = True
				print ("too fast linear speed")
			elif(self.v < -self.phi_max*self.r):
				self.v = -self.phi_max*self.r
				self.wrong_speed = True
				print ("too fast linear speed")

			if(self.v == 0):
				if(self.w == 0):
					beta = .0
				else:
					beta = 1000000 # just impossible
			else:
				beta = math.atan(self.w*self.d/self.v)
			
			if (beta>self.beta_max):
				self.w = self.v*math.tan(self.beta_max)/self.d
				self.wrong_speed = True
				print ("too fast turn")
			elif (beta < -self.beta_max):
				self.w = self.v*math.tan(-self.beta_max)/self.d
				self.wrong_speed = True
				print ("too fast turn")


		# print (self.v, self.w)


# Destination is stored in a list
class E160_des_state:
	def __init__(self):
		self.x = []
		self.y = []
		self.theta = []
		self.p = 0 #current goal in the list
		

	def reset_destination(self,x,y,theta):
		self.x = []
		self.y = []
		self.theta = []
		self.p = 0
		print("destinations has been reset")
		self.add_destination(x,y,theta)



	def add_destination(self,x,y,theta):
		self.x.append(x)
		self.y.append(y)
		self.theta.append(theta)
		print(str(x)+" "+str(y)+" "+str(theta)+"  is added to destination list")


	def get_des_size(self):
		return len(self.x)


	def get_des_state(self):
		return self.x[self.p], self.y[self.p], self.theta[self.p]

	def reach_destination(self):
		self.p += 1
		if(self.p < len(self.x)):
			return False #reached a midway destination
		else:
			return True #reached to the final destination
