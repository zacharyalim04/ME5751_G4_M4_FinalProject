import heapq #this allows us to use python lists in a priority queue configuration, from my reserach this is one of the most popular methods in python
import cv2
import numpy as np
import math
import warnings
from PIL import Image, ImageTk
from PIL.DdsImagePlugin import item2

from bsTree import *
from Path import *
# from Queue import Queue


class path_planner:
	def __init__(self,graphics):


		self.graphics = graphics
		# self.graphics.scale = 400 #half pixel number on canvas, the map should be 800 x 800
		# self.graphics.environment.robots[0].set_bot_size(body_cm = 2*self.inflation_radius)
		#self.graphics.environment.width/height = 2

		self.costmap = self.graphics.map
		self.map_width = self.costmap.map_width
		self.map_height = self.costmap.map_height

		self._init_path_img()
		self.path = Path()

		# HEURISTIC SECTION
		# there are multiple ways we can make a heuristic, but we will use euclidean which is basically just the straight line distance to the goal
		self.heuristic_type = "euclidean"

		# set heuristic weighting, basically, how much does the path planner care about the heuristic, will it chase it heavily (greedy A*) or not care at all (dijsktra)
		self.mu = 1  # 1 for a star, 0 for dijkstra, 1> for greedy

		#to show the live path
		self.live_search_vis = True
		self.live_update_rate = 200 #how many nodes will it update the live tracking on the gui

		self.set_start(world_x = 0, world_y = 0)
		self.set_goal(world_x = 179.0, world_y = -230.0, world_theta = .0)

		self.plan_path()
		self._show_path()


	def _flush_gui(self):
		try:
			self.graphics.canvas.update_idletasks()
			self.graphics.canvas.update()
		except Exception:
			pass

	def _heuristic(self, node, goal): #function used to calculate the heuristic, it takes in the node and goal
		#gets the points coordinates for the nodes and goal so we can use it to calculate the straight line distance
		i1 = node[0]
		j1 = node[1]
		i2 = goal[0]
		j2 = goal[1]

		dx = abs(i1 - i2) #make sure to use the magnitude of the distance
		dy = abs(j1 - j2)
		if self.heuristic_type == "euclidean":
			#pythagorean theorem to return the hypotenuse
			return math.sqrt(dx * dx + dy * dy)

		elif self.heuristic_type == "manhattan": #this reduces the heuristic to finding the closest path by the manhattan search, so like our cost map
			return dx + dy


	def _neighbors(self, node): #this function is used to find the neighbors, remember we are searching for 8 neighbors

		i = node[0] #make it less confusing to unpack now
		j = node[1]

		neighbors_list = [] #create an empty list to store our neighbors in

		#when di  = -1, and dj = -1 , we are looking at the top left, di = -1, dj = 0, we are looking up, di = 0, dj = 0, current node, di = 1 , dj = 0, looking down, etc etc.....

		for di in [-1, 0, 1]:
			for dj in [-1, 0, 1]:
				if di == 0 and dj == 0: #this is because at 0,0 that is our current node
					continue #so we just skip it
				neighbor_i = i + di #updates the row coordinate of the neighbor, so if we are at i = 100, then we are looking up, it would be 100 + -1 to give the y coordinate of the node
				neighbor_j = j + dj #for each point, update the neighbor same here

				#make sure the neighbor is inside the actual map before adding it into the neighbors list

				if (0 <= neighbor_i < self.map_height) and (0 <= neighbor_j < self.map_width): #just checks it in the bounds
					neighbors_list.append((neighbor_i, neighbor_j)) #appends it to our neighbors list if it is within the bounds

		return neighbors_list

	def _get_move_cost(self, current_node, neighbor_node): #this function is used to get the movement cost between nodes, remember, we need to explore the path that is the cheapest first
		obstacle_cost_threshold = 254 #make sure this is the same as from cost_map, ensures we never go into an obstacle node

		i2, j2 = neighbor_node #takes out the vi and j values of the neighbor node
		costmap_value = self.costmap.costmap[i2, j2] #takes the cost of the pixel at a certain i and j, then if it's an obstacle, we cant use it

		if costmap_value > obstacle_cost_threshold:
			return float('inf') #makes the obstacles so expensive so that it never goes there, better to use infinity instead of a very high number in case there ever is a path that is longer than say 999999999999, then the algorithm  breaks

		#checks to see if the anticipated movement is diagonal, if both of these values the neighboring values are not equal to the current, then it must be diagonal,
		#for example, take i1 = 0, j1 = 0, if we go diagonal it is always either up right, up left, down right, down left, so both values have to have changed to be diagonal
		i1, j1 = current_node

		if i1 != i2 and j1 != j2:
			is_diagonal = True
		else:
			is_diagonal = False

		if is_diagonal:
			distance_cost = 1.414 #using basic geometry, going diagonal = sqrt2
		else:
			distance_cost = 1.0 #otherwise it's just a translation, up left down or right

		return distance_cost + costmap_value #what is the cost of the distance moved and the cost at the map but doesn't account for heuristic

	def _get_final(self, origin_dict, current_node): #uses the dictionary that has the origin of the current node, where did it come from

		#logic is that the current node will be seen as the "end" or "goal" node, then you build the path backwards
		reversed_path = []
		#then, going backwards through the path up to that node to build the path
		#the data structure will be a dictionary, since it can store a key:value, which will hold the node and its closest neighbor and so forth
		while current_node in origin_dict: #go through every node
			reversed_path.append(current_node) #adds to the path list, the current load

			#then we will use the dictionary to find the node we came from at that current node and update it
			current_node = origin_dict[current_node]
		reversed_path.append(current_node) #make sure to include the end node after running through the dictionary, this will be the start node, but it's not counted in for while loop

		final_path = list(reversed(reversed_path))  #reverses the reversed path so that way it flows in the proper direction since we started populating it backwards

		print(f"Path found successfuly. Total poses: {len(final_path)}")


		#now we have to add this to the poses
		total_path_length_pixels = 0.0
		last_pose_coords = None

		for pose_coords in final_path:

			#break this up from a touple otherwise it gets super annoying later in the code
			map_i, map_j = pose_coords

			#not necessary to calculate theta, but good for later use in our project so why not do it now
			current_theta = 0.0
			if last_pose_coords is not None:
				last_i, last_j = last_pose_coords
				#use geometry (same thing from our P controller code) to find the angle between points

				current_theta = math.atan2(map_j - last_j, map_i - last_i)

			#adds the pose to the path
			self.path.add_pose(Pose(map_i=map_i, map_j=map_j, theta=current_theta))

			#calculate the total path length
			if last_pose_coords is not None:
				di = map_i - last_pose_coords[0]
				dj = map_j - last_pose_coords[1]
				total_path_length_pixels += math.sqrt(di ** 2 + dj ** 2)

			last_pose_coords = pose_coords #saves current coords as last point so the loop runs without issue and the next loop calculations can be found
		print(f"Total path length: {total_path_length_pixels:.2f} pixels")



	def set_start(self, world_x = 0, world_y = 0, world_theta = 0):
		self.start_state_map = Pose()
		map_i, map_j = self.world2map(world_x,world_y)
		print("Start with %d, %d on map"%(map_i,map_j))
		self.start_state_map.set_pose(map_i,map_j,world_theta)


	def set_goal(self, world_x, world_y, world_theta = 0):
		self.goal_state_map = Pose()
		map_i, map_j = self.world2map(world_x, world_y)
		print ("our new goal is %d, %d on map"%(map_i,map_j))
		self.goal_state_map.set_pose(map_i, map_j, world_theta)


	#convert a point a map to the actual world position
	def map2world(self,map_i,map_j):
		world_x = -self.graphics.environment.width/2*self.graphics.scale + map_j
		world_y = self.graphics.environment.height/2*self.graphics.scale - map_i
		return world_x, world_y

	#convert a point in world coordinate to map pixel
	def world2map(self,world_x,world_y):
		map_i = int(self.graphics.environment.width/2*self.graphics.scale - world_y)
		map_j = int(self.graphics.environment.height/2*self.graphics.scale + world_x)
		if(map_i<0 or map_i>=self.map_width or map_j<0 or map_j>=self.map_height):
			warnings.warn("Pose %f, %f outside the current map limit"%(world_x,world_y))

		if(map_i<0):
			map_i=int(0)
		elif(map_i>=self.map_width):
			map_i=self.map_width-int(1)

		if(map_j<0):
			map_j=int(0)
		elif(map_j>=self.map_height):
			map_j=self.map_height-int(1)

		return map_i, map_j

	def _init_path_img(self):
		self.map_img_np = 255*np.ones((int(self.map_width),int(self.map_height),4),dtype = np.int16)
		# self.map_img_np[0:-1][0:-1][3] = 0
		self.map_img_np[:,:,3] = 0

	def _show_path(self):
		for pose in self.path.poses:
			map_i = pose.map_i
			map_j = pose.map_j
			self.map_img_np[map_i][map_j][0] =255 #red
			self.map_img_np[map_i][map_j][1] =0 #blue
			self.map_img_np[map_i][map_j][2] =0 #green
			self.map_img_np[map_i][map_j][3] =255 #opacity

		self.path_img=Image.frombytes('RGBA', (self.map_img_np.shape[1],self.map_img_np.shape[0]), self.map_img_np.astype('b').tostring())
		self.graphics.draw_path(self.path_img)



		# If you want to save the path as an image, un-comment the following line:
		# self.path_img.save('Log\path_img.png')

		# If you want to output an image of map and path, un-comment the following two lines
		# self.path_img = toimage(self.map_img_np)
		# self.path_img.show()

	def plan_path(self):
		print(f"Planning path with mu = {self.mu}")

		#clear previous path in case this code was ran multiple times
		self.path.clear_path()

		self._init_path_img()

		#just initializes our start and goal form previously defined functions
		start_node = (self.start_state_map.map_i, self.start_state_map.map_j)
		goal_node = (self.goal_state_map.map_i, self.goal_state_map.map_j)


		#we will be using the priority queue here, we will use heapq which will always pop the lowest value in the list for us

		open_list_pq = []

		origin_dict = {} #stores the child node: parent node, so that way we can use it to save our path

		g_score_map = {} #the actual cost of simply going from the starting node to any oter node, stored as dictionary to have (start coord, cost)

		g_score_map[start_node] = 0.0 #initializes the g_score for the start node
		g_start = g_score_map[start_node]

		h_start = self._heuristic(start_node, goal_node) #gets the heuristic and defines it at the start

		#remember our cost is g + h

		f_start = g_start + self.mu * h_start #the total cost is the actual cost plus the heuristic cost, so if mu is zero, we'd essentially have dijsktra

		nodes_explored = 0 #counts how many nodes we explored, as per the rubric

		closed_set = set() #basically the same as a list, but is just used for checking if something is in this list, we will add nodes that are already searched through here
		#this isn't a python class, so I don't think we need to explain why we used a closed set, but when I was testing the algorithm it was slower using a list, found out people use sets
		#https://stackoverflow.com/questions/2831212/python-sets-vs-lists reference

		#used this for heap https://docs.python.org/3/library/heapq.html

		heapq.heappush(open_list_pq, (f_start, h_start, start_node))
		#open_list_pq is the list we add to, heappush means we are adding to the priority queue
		#first, it will sort by f_start and then use h_start as a tiebreaker, then the data it stores at that location is start_node

		#now we loop through the open list

		while open_list_pq:
			#get the lowest f_score from queue first,

			current_f, current_h, current_node = heapq.heappop(open_list_pq) #takes our smallest item out of the priority queue and unpacks it into our current  values


			if current_node in closed_set:
				continue #if we've already processed it, then just skip

			nodes_explored += 1
			if current_node == goal_node:
				print(f"Goal reached, exploring {nodes_explored} nodes")

				#now build final path

				self._get_final(origin_dict, current_node)

				#save the path to a file
				self.path.save_path(file_name='Log\\path.csv')
				return #terminates loop if we reached the goal

			closed_set.add(current_node) #if it wasn't the goal, add it to the closed set

			if self.live_search_vis:
				i, j = current_node

				#this is used to show the live tracking
				self.map_img_np[i, j, 0] = 0 #red
				self.map_img_np[i, j, 1] = 0 #green
				self.map_img_np[i, j, 2] = 255 #blue
				self.map_img_np[i, j, 3] = 255 #opacity

			#update gui every couple hundred nodes or so

			if (nodes_explored % self.live_update_rate) == 0:

				self.path_img = Image.frombytes('RGBA', (self.map_img_np.shape[1], self.map_img_np.shape[0]), self.map_img_np.astype('b').tostring())
				#code to make the path it is currently looking at pop up as red
				path_nodes_to_reset = []
				trace_node = current_node
				while trace_node in origin_dict:
					path_nodes_to_reset.append(trace_node)
					trace_node = origin_dict[trace_node]

				#draw current path in red
				for i, j in path_nodes_to_reset:
					self.map_img_np[i, j, 0] = 255
					self.map_img_np[i, j, 1] = 0
					self.map_img_np[i, j, 2] = 0
					self.map_img_np[i, j, 3] = 255
				self.path_img = Image.frombytes('RGBA', (self.map_img_np.shape[1],self.map_img_np.shape[0]), self.map_img_np.astype('b').tostring())
				self.graphics.draw_path(self.path_img)
				self._flush_gui()

				#after it's searched, if its not the best path, revert pixels to blue
				for i, j in path_nodes_to_reset:
					self.map_img_np[i, j, 0] = 0
					self.map_img_np[i, j, 1] = 0
					self.map_img_np[i, j, 2] = 255

			#then check all of our neighbors from the current goal

			for neighbor in self._neighbors(current_node):

				if neighbor in closed_set:
					continue #skip if processed

				move_cost = self._get_move_cost(current_node, neighbor)

				if move_cost == float('inf'):
					continue #skip any obstacles/unreachable nodes


				current_g_score = g_score_map.get(current_node, float('inf')) + move_cost
				#update the g score if it's less than any found path until now
				if current_g_score < g_score_map.get(neighbor, float('inf')): #if we haven't seen it before, assign it infinity

					origin_dict[neighbor] = current_node #if it is, then update it so that way the parent node gets updated for this child node to be the cheapest path
					g_score_map[neighbor] = current_g_score

					#calculate heuristic
					h_score = self._heuristic(neighbor, goal_node)
					f_score = current_g_score + self.mu * h_score

					#then add the neighbor to the priority queue
					heapq.heappush(open_list_pq, (f_score, h_score, neighbor))
		print(f'No possible paths found after exploring {nodes_explored} nodes')






# bresenham algorithm for line generation on grid map
# from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
def bresenham(x1, y1, x2, y2):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions

    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()

    # print points
    return points
