import cv2
import numpy as np
import math
from PIL import Image, ImageTk
from queue import Queue


class cost_map:
    def __init__(self, graphics):
        self.graphics = graphics

        self.inflation_radius = 18  # radius of our robot is 18 pixel or cm
        self.graphics.environment.robots[0].set_bot_size(body_cm=2 * self.inflation_radius)

        self.map_width = int(self.graphics.environment.width * self.graphics.scale)
        self.map_height = int(self.graphics.environment.height * self.graphics.scale)

        # To adjust the map size, please go to E160_graphics.py line 18, and change the self.scale there.
        # self.scale shall be half of your image size, for example, if your map image is 500 x 500 pixel, set your self.scale = 250
        # The default map size is 500 x 500 pixel. In case you want a smaller size for debugging, you can change the value of self.scale.

        try:
            self.load_map(map=f"maps/mapC_wide_occ.png")  # load map, put your own map here
        except:
            self.graphics.show_map_button.configure(state="disabled")
            print("no map loaded")  # if fail to find the map png
            return

        self.show_map()
        self.compute_costmap()
        self.get_vis_map()
        self.save_vis_map(map="maps/mapC_wide_occ_vis.png")
        self.save_costmap(file_path='maps/mapC_wide_occ_cost.txt')

    # load occupancy grid into self.map
    # self.map is a numpy 2d array
    # initialize self.costmap, a numpy 2d array, same as self.map
    def load_map(self, map="maps/mapC_wide_occ.png"):
        self.map_img = Image.open(map).convert('L')
        self.map_img = self.map_img.resize((int(self.map_width), int(self.map_height)))
        # self.graphics.draw_map(map_img=self.map_img)
        self.map = cv2.imread(map, cv2.IMREAD_GRAYSCALE)
        print(self.map.dtype)
        print("Loaded map dimension: %d x %d pixel" % (self.map.shape[0], self.map.shape[1]))
        self.map = cv2.resize(self.map, dsize=(int(self.map_width), int(self.map_height)),
                              interpolation=cv2.INTER_CUBIC)

        self.distmap = np.full_like(self.map, -1,
                                    dtype=int)  # map for saving distance to the nearest obstacle, initialize with an array with all -1

        self.costmap = np.copy(self.map).astype(float)  # the cost map you are going to work on
        self.vis_map = np.copy(self.map)  # map for visualization, intialize same as the map

    # save your costmap into a grayscale image
    def save_vis_map(self, map="maps/mapC_wide_occ_vis.png"):
        save_img = Image.fromarray(self.vis_map)
        save_img.save(map)

    def show_vis_map(self):
        self.vis_map_img = Image.frombytes('L', (self.vis_map.shape[1], self.vis_map.shape[0]),
                                           self.vis_map.astype('b').tostring())
        self.graphics.draw_map(map_img=self.vis_map_img)

    # display costmap on the dialogue window, in most cases, the cost map cannot be properly shown
    def show_costmap(self):
        self.costmap_img = Image.frombytes('L', (self.costmap.shape[1], self.costmap.shape[0]),
                                           self.costmap.astype('b').tostring())
        self.graphics.draw_map(map_img=self.costmap_img)

    # display binary occupancy grid on the dialogue window
    def show_map(self):
        self.graphics.draw_map(map_img=self.map_img)

    def save_costmap(self, file_path, fmt='%.3f', delimiter='\t'):
        np.savetxt(file_path, self.costmap, fmt=fmt, delimiter=delimiter)

    def compute_costmap(self):
        # we haven't explored anything, so the cost map should be unexplored first
        self.distmap[:] = -1

        # define thresholds...due to compression in the image, black pixels may appear gray
        obstacle_th = 20
        free_th = 220

        self.distmap[self.map < obstacle_th] = 0

        # we need to scan the map/array and collect the values and mark it as either obstacle or free space.
        height, width = self.map.shape  # assigns height and width of the array to the size of the map using the shape method in numpy

        # queue
        q = Queue()  # this is for the breadth first search, this is first in first out

        for h in range(height):
            for w in range(width):
                current_value = int(
                    self.map[h, w])  # basically takes the current value at any given h,w point in the map
                if current_value < obstacle_th:
                    self.distmap[
                        h, w] = 0  # set the value of the point that is an obstacle to zero, because the distance to that obstacle is zero
                    q.put((h, w))

        # so far, we took the map and marked all the obstacles and set their distance values to zero
        neighbors = [(1, 0), (-1, 0), (0, 1),
                     (0, -1)]  # need to define a set of neighbors so that way we can check each opint
        while not q.empty():
            h, w = q.get()

            # take our current point
            d = self.distmap[h, w]  # this gives us our current point

            for dy, dx in neighbors:
                ny = h + dy  # neighbor height
                nx = w + dx  # neighbor width

                if ny < 0 or ny >= height:  # this code makes sure it doesn't search out of bounds
                    continue
                if nx < 0 or nx >= width:
                    continue

                if self.distmap[ny, nx] != -1:  # if its not unexplored, continue
                    continue


                # if it isn't an obstacle or is unvisited, then we continue to map this location
                self.distmap[ny, nx] = d + 1  # the original point +1
                q.put((ny,
                       nx))  # adds this point into the queue to search from later, so we will revisit this point and then map its neighbors and so on

        # previously, we built a distance map, now we need to use this distance map to make a cost map

        obstacle = 255.0
        unknown = 75
        decay = 18
        self.costmap[
            :, :] = 0.0  # sets every cell in the COST map to 0, so we can then use the distance map to assign costs

        for h in range(height):
            for w in range(width):
                d = int(self.distmap[h, w])  # gets the distance to the nearest obstacle in this specific location
                if d == 0:
                    self.costmap[
                        h, w] = obstacle  # since the only cell with distance 0 is an obstacle itself, we make it an obstacle
                elif d == -1:
                    self.costmap[h, w] = unknown
                elif d <= int(self.inflation_radius):
                    self.costmap[
                        h, w] = obstacle  # essentially, if the distance to the nearest wall is less than or equal to the inflation radius, we mark it as an obstacle too
                else:
                    dist_beyond = d - int(self.inflation_radius)

                    t = dist_beyond / float(max(1,
                                                decay))  # the distance beyond the inflation walls becomes a smaller, basically controls the decay factor
                    if t < 0.0:
                        t = 0.0
                    if t > 1.0:
                        t = 1.0

                    gamma = 0.6
                    factor = (1.0 - t) ** gamma

                    tapered = unknown * factor
                    if tapered >= obstacle:
                        tapered = obstacle - 1.0
                    if tapered < 0.0:
                        tapered = 0.0
                    self.costmap[h, w] = tapered + 1
        '''
#make it binary, all pixels should be obstacle or unknown, since png compresses black pixels and they may become lighter
		#self.costmap is initialized same as the map. That is, a white pixel is 255.0 (free), a black pixel is initalized as 0.0 (occupied). 

		#We first let all black obstacles to be 0 on distmap
		self.distmap[self.map<20.0] = 0
		print("there are: "+str(np.sum(self.distmap==0))+ " pixels contain obstacle")

		#It is up cto you how to get distmap, you may choose not to use it even

		#just a demo how you generate a costmap, we believe any obstacle pixel (distvalue = 0) has a cost of 255
		#we also believe all other pixels shall have cost of 128
		#this is where we actually do the code to make the cost map
		self.costmap[self.distmap==0]=255.0
		self.costmap[self.distmap!=0]=128.0
		pass
'''

    # scale costmap to 0 - 255 for visualization
    def get_vis_map(self):
        '''
		get the map for visualization
		'''
        min_val = min(np.min(self.costmap), 0)
        max_val = np.max(self.costmap)

        # Normalize the costmap
        self.vis_map = np.uint8(255 - (self.costmap - min_val) / (max_val - min_val) * 255)