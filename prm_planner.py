import cv2
import numpy as np
import math
import random
from PIL import Image, ImageTk
from ackermann_path import curvature_of_fillet, interp_line
from scipy.spatial import KDTree
from Path import *
# from Queue import Queue

class prm_node:
    def __init__(self,map_i=int(0),map_j=int(0)):
        self.map_i = map_i
        self.map_j = map_j
        self.edges = [] #edges of child nodes
        self.parent = None #parent node


class prm_edge:
    def __init__(self,node1=None,node2=None):
        self.node1 = node1 #parent node
        self.node2 = node2 #child node

#You may modify it to increase efficiency as list.append is slow
class prm_tree:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_nodes(self,node):
        self.nodes.append(node)

    #add an edge to our PRM tree, node1 is parent, node2 is the kid
    def add_edges(self,node1,node2): 
        edge = prm_edge(node1,node2)
        self.edges.append(edge)
        node1.edges.append(edge)
        node2.parent=edge.node1


class path_planner:
    def __init__(self,graphics):
        self.graphics = graphics

        self.costmap = self.graphics.map
        self.map_width = self.costmap.map_width
        self.map_height = self.costmap.map_height

        self.pTree=prm_tree()

        self._init_path_img()
        self.path = Path()

        self.min_turn_radius = 40.0  # pixels; tune for your 4-wheel Ackermann vehicle
        # Start and goal will be set later from the GUI callbacks
        self.start_node = None
        self.goal_node = None


    def set_start(self, world_x = 0, world_y = 0, world_theta = 0):
        self.start_state_map = Pose()
        map_i, map_j = self.world2map(world_x,world_y)
        print("Start with %d, %d on map"%(map_i,map_j))
        self.start_state_map.set_pose(map_i,map_j,world_theta)
        self.start_node = prm_node(map_i,map_j)
        self.pTree.add_nodes(self.start_node)

    def set_goal(self, world_x, world_y, world_theta = 0):
        self.goal_state_map = Pose()
        goal_i, goal_j = self.world2map(world_x, world_y)
        print ("goal is %d, %d on map"%(goal_i, goal_j))
        self.goal_state_map.set_pose(goal_i, goal_j, world_theta)
        self.goal_node = prm_node(goal_i,goal_j)
        self.pTree.add_nodes(self.goal_node)

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
            Warning.warn("Pose %f, %f outside the current map limit"%(world_x,world_y))

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
            self.map_img_np[map_i][map_j][1] =0
            self.map_img_np[map_i][map_j][2] =0
            self.map_img_np[map_i][map_j][3] =255

        np.savetxt("file.txt", self.map_img_np[1])

        self.path_img=Image.frombytes('RGBA', (self.map_img_np.shape[1],self.map_img_np.shape[0]), self.map_img_np.astype('b').tostring())
        # self.path_img = toimage(self.map_img_np)
        #self.path_img.show()
        self.graphics.draw_path(self.path_img)

    def check_vicinity(self,x1,y1,x2,y2,threshold = 1.0): #can MODIFY THIS CODE HERE AND CHANGE HOWEVER,
        if(math.sqrt((x1-x2)**2+(y1-y2)**2)<threshold):
            return True
        else:
            return False

    ## Ensure line does not go through obstacles
    def _line_is_free(self, n1, n2):
        points = bresenham(n1.map_i, n1.map_j, n2.map_i, n2.map_j)
        for (i, j) in points:
            if self.costmap.costmap[i][j] >= 255:
                return False
        return True

    ## Breadth-First Search from Start to Goal
    def _bfs_search(self, start, goal):
        from collections import deque

        queue = deque([start])
        visited = set([start])
        parent = {start: None}

        nodes_searched = 0 # How many Nodes were SEARCHED

        while queue:
            curr = queue.popleft()
            nodes_searched += 1

            if curr == goal:
                break
            
            for e in curr.edges:
                nxt = e.node2
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = curr
                    queue.append(nxt)
        print(f"Nodes Searched: {nodes_searched}")

        if goal not in parent:
            return None

        # Reconstruct Path
        path = []
        node = goal
        while node:
            path.append(node)
            node = parent[node]
        return list(reversed(path))

    # Create shortcut if shortcut exists
    def _shortcut_path(self, nodes):
        if not nodes:
            return nodes
        
        new_path = [nodes[0]]
        i = 0
        while i < len(nodes) - 1:
            j = len(nodes) - 1
            while j > i + 1:
                if self._line_is_free(nodes[i], nodes[j]):
                    new_path.append(nodes[j])
                    i = j
                    break
                j -= 1
            else:
                new_path.append(nodes[i + 1])
                i += 1

        return new_path        

    def _remove_short_segments(self, nodes, min_dist=20):
        """
        Removes consecutive nodes that are too close together.
        nodes: list of prm_node in sequence
        min_dist: minimum allowed distance in pixels
        Returns: pruned list of nodes
        """
        if not nodes:
            return nodes

        new_nodes = [nodes[0]]
        last = nodes[0]

        for n in nodes[1:]:
            dx = n.map_i - last.map_i
            dy = n.map_j - last.map_j
            dist = math.hypot(dx, dy)

            if dist >= min_dist:
                new_nodes.append(n)
                last = n

        return new_nodes

    def _remove_sharp_angles(self, nodes, min_angle_deg=25):
        """
        Removes nodes that create extremely sharp turning angles.
        Keeps start and goal.
        """
        if len(nodes) <= 2:
            return nodes

        new_nodes = [nodes[0]]
        for i in range(1, len(nodes) - 1):
            p_prev = nodes[i - 1]
            p = nodes[i]
            p_next = nodes[i + 1]

            v1 = (p.map_i - p_prev.map_i, p.map_j - p_prev.map_j)
            v2 = (p_next.map_i - p.map_i, p_next.map_j - p.map_j)

            L1 = math.hypot(v1[0], v1[1])
            L2 = math.hypot(v2[0], v2[1])
            if L1 < 1e-6 or L2 < 1e-6:
                continue

            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (L1 * L2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))

            # If angle is too sharp → skip this node
            if angle < min_angle_deg:
                continue

            new_nodes.append(p)

        new_nodes.append(nodes[-1])
        return new_nodes

    def _remove_collinear(self, nodes, angle_tol_deg=10):
        """
        Removes nodes that lie nearly collinear between neighbors.
        Useful to eliminate stair-step patterns along obstacle boundaries.
        """
        if len(nodes) <= 2:
            return nodes

        new_nodes = [nodes[0]]

        for i in range(1, len(nodes) - 1):
            p_prev = nodes[i - 1]
            p = nodes[i]
            p_next = nodes[i + 1]

            v1 = (p.map_i - p_prev.map_i, p.map_j - p_prev.map_j)
            v2 = (p_next.map_i - p.map_i, p_next.map_j - p.map_j)

            L1 = math.hypot(v1[0], v1[1])
            L2 = math.hypot(v2[0], v2[1])
            if L1 < 1e-6 or L2 < 1e-6:
                continue

            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (L1 * L2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))

            # If angle is almost straight (collinear) → remove this node
            if abs(angle) < angle_tol_deg:
                continue

            new_nodes.append(p)

        new_nodes.append(nodes[-1])
        return new_nodes


    def _compute_thetas(self, pts):
        """
        pts: list of (map_i, map_j) points (pixel coords)
        returns: list of (map_i, map_j, theta) with heading angles
        """
        out = []
        n = len(pts)
        for i in range(n):
            if i < n - 1:
                dx = pts[i+1][1] - pts[i][1]   # map_j is x
                dy = pts[i+1][0] - pts[i][0]   # map_i is y
            else:
                dx = pts[i][1] - pts[i-1][1]
                dy = pts[i][0] - pts[i-1][0]

            theta = math.atan2(-dy, dx)
            out.append((pts[i][0], pts[i][1], theta))

        # Optional heading smoothing to remove discontinuities
        for i in range(1, n-1):
            prev_theta = out[i-1][2]
            next_theta = out[i+1][2]

            # Weighted average
            smoothed = math.atan2(
                math.sin(prev_theta) + math.sin(next_theta),
                math.cos(prev_theta) + math.cos(next_theta)
            )

            out[i] = (out[i][0], out[i][1], smoothed)

        return out
        

    def plan_path(self):
        # Set start from current robot pose
        robot = self.graphics.environment.robots[0]
        rx, ry, rtheta = robot.state.get_pos_state()
        self.set_start(world_x=rx, world_y=ry, world_theta=rtheta)

        if self.goal_node is None:
            print("No goal set yet; right-click to choose a goal first.")
            return
        # Retry until a valid path is found
        max_attempts = 50
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print(f"Planning attempt {attempt}")

            # Reset the PRM tree for this attempt
            self.pTree = prm_tree()
            self.pTree.add_nodes(self.start_node)
            self.pTree.add_nodes(self.goal_node)


            ## Define other variables needed for path planning
            num_samples = 300 # Number of Random Nodes to create
            K_neighbors = 10 # Number of connections per node
            nodes = self.pTree.nodes # Start and goal already in here

            self.path.clear_path() # Clear old path

            nodes_generated = 0 # How many nodes were CREATED
            
            ## Generate random samples
            for _ in range(num_samples):
                if random.random() < 0.7:
                    # Sample in a shrinking radius around goal
                    radius = 10   # TUNE THIS RADIUS!!!
                    gi, gj = self.goal_node.map_i, self.goal_node.map_j

                    ri = int(gi + random.randint(-radius, radius))
                    rj = int(gj + random.randint(-radius, radius))

                    # Set Map Boundaries
                    if ri < 0: ri = 0
                    if ri >= self.map_width: ri = self.map_width - 1
                    if rj < 0: rj = 0
                    if rj >= self.map_height: rj = self.map_height - 1
                else:
                    # regular uniform sampling
                    ri = random.randint(0, self.map_width-1)
                    rj = random.randint(0, self.map_height-1)

                # Skip obstacles
                if self.costmap.costmap[ri][rj] >= 255:
                    continue

                nodes.append(prm_node(ri, rj))
                nodes_generated += 1 

            # Boundary Sampling for narrow corridors
            boundary_samples = 400  # TUNE THE BOUNDARY SAMPLES
            for _ in range(boundary_samples):

                # pick a random pixel
                ri = random.randint(1, self.map_width - 2)
                rj = random.randint(1, self.map_height - 2)

                # skip if obstacle
                if self.costmap.costmap[ri][rj] >= 255:
                    continue

                # check if neighbor is obstacle → then ri,rj is ON boundary
                if (self.costmap.costmap[ri+1][rj] >= 255 or
                    self.costmap.costmap[ri-1][rj] >= 255 or
                    self.costmap.costmap[ri][rj+1] >= 255 or
                    self.costmap.costmap[ri][rj-1] >= 255):

                    nodes.append(prm_node(ri, rj))
                    nodes_generated += 1
            
            # Narrow-Passage Sampling (Inflation-difference method)
            inflation = 3  # tune based on passage width
            narrow_samples = 200

            # Create a slightly inflated binary obstacle map
            kernel = np.ones((inflation, inflation), np.uint8)
            obs = (self.costmap.costmap >= 255).astype(np.uint8)
            inflated = cv2.dilate(obs, kernel)

            # Regions that appear ONLY in inflated map are narrow passage borders
            narrow_mask = (inflated == 1) & (obs == 0)

            # Collect candidate coordinates
            narrow_coords = np.argwhere(narrow_mask)

            for _ in range(narrow_samples):
                if len(narrow_coords) == 0:
                    break
                idx = random.randint(0, len(narrow_coords)-1)
                mi, mj = narrow_coords[idx]

                # Double check: ensure free space
                if self.costmap.costmap[mi][mj] >= 255:
                    continue

                nodes.append(prm_node(mi, mj))
                nodes_generated += 1

            print(f"Nodes Generated: {nodes_generated}")

            ## KD-Tree Implementation
            node_points = [(n.map_i, n.map_j) for n in nodes]
            kdtree = KDTree(node_points)

            ## Connect nearest neighbors USING KD-Tree
            for idx, node in enumerate(nodes): # Sets loop to occur for all Nodes created

                # Set the KD-Tree Query size to the neighboring nodes + 1 (The node itself)
                dists, neighbor_indices = kdtree.query(node_points[idx], k=K_neighbors+1)

                # Always output a LIST, so that the loops can iterate
                if K_neighbors+1 == 1:
                    neighbor_indices = [neighbor_indices]
                else:
                    # ensure it's a list, not a numpy scalar
                    neighbor_indices = list(np.atleast_1d(neighbor_indices))

                # KD-Tree includes the initial Node itself, so skip that one
                # The self-Node is always first since the distance is zero
                neighbor_indices = neighbor_indices[1:]

                # For each neighbor, add an edge if the line connecting the Nodes is unobstructed
                for ni in neighbor_indices:
                    nb = nodes[ni]
                    if self._line_is_free(node, nb):
                        self.pTree.add_edges(node, nb)
            
            ## Breadth-First Search through generated Roadmap
            path_nodes = self._bfs_search(self.start_node, self.goal_node)

            # If BFS failed → retry
            if not path_nodes:
                print("No path found, retrying...")
                # Reset roadmap before next attempt
                self.pTree = prm_tree()
                self.pTree.add_nodes(self.start_node)
                self.pTree.add_nodes(self.goal_node)
                continue  # restart loop

            # Use shortcut to remove unnecessary nodes in list
            path_nodes = self._shortcut_path(path_nodes)

            # If shortcut empties path (rare), retry
            if not path_nodes:
                print("Shortcut collapsed path, retrying...")
                self.pTree = prm_tree()
                self.pTree.add_nodes(self.start_node)
                self.pTree.add_nodes(self.goal_node)
                continue

            # Remove very short PRM segments to prevent tiny turns
            path_nodes = self._remove_short_segments(path_nodes, min_dist=20)

            path_nodes = self._remove_sharp_angles(path_nodes, min_angle_deg=25)

            path_nodes = self._remove_collinear(path_nodes, angle_tol_deg=10)

            ## Draw pixel path using Node Roadmap (with Ackermann-compliant fillets)
            raw_points = [(n.map_i, n.map_j) for n in path_nodes]

            def is_free(i, j):
                if i < 0 or i >= self.map_width or j < 0 or j >= self.map_height:
                    return False
                return self.costmap.costmap[i][j] < 255

            # Generate smoothed path with circular fillets at corners
            # --- Ackermann Fillet Path Generation ---
            smooth_points = []
            prev = raw_points[0]
            smooth_points.append(prev)

            for i in range(1, len(raw_points)-1):
                p_prev = raw_points[i-1]
                p = raw_points[i]
                p_next = raw_points[i+1]

                fillet = curvature_of_fillet(p_prev, p, p_next, R_min=self.min_turn_radius)

                if fillet is None:
                    # straight segment only
                    smooth_points.extend(interp_line(prev, p, step=0.5))
                    prev = p
                    continue

                t1, t2, c, R, th1, th2 = fillet

                # straight into fillet
                smooth_points.extend(interp_line(prev, t1, step=0.5))

                # arc
                arc_len = abs(th2 - th1) * R
                N = max(int(arc_len / 0.5), 1)
                for k in range(N + 1):
                    th = th1 + (th2 - th1) * k / N
                    x = c[0] + R * math.cos(th)
                    y = c[1] + R * math.sin(th)
                    smooth_points.append((x, y))

                prev = t2

            # ⬇⬇⬇ ADD THIS AFTER THE LOOP ⬇⬇⬇
            # final segment to last waypoint
            smooth_points.extend(interp_line(prev, raw_points[-1], step=0.5))


            # Compute heading angles for the smoothed path
            pts_with_theta = self._compute_thetas(smooth_points)

            # Write smoothed path with proper orientation
            for pi, pj, th in pts_with_theta:
                self.path.add_pose(
                    Pose(map_i=int(round(pi)), map_j=int(round(pj)), theta=th)
                )


            # Save a thinned version of the path for the controller
            out_path = "Log/prm_path.csv"
            with open(out_path, "w") as f:
                # header
                f.write("map_i\tmap_j\ttheta\n")

                # keep only every Nth point so we don't overload the controller/GUI
                N = 30  # TUNE THIS NTH VALUE FOR LESS WAYPOINTS

                for k, (pi, pj, th) in enumerate(pts_with_theta):
                    # Keep every Nth point and always keep the last one
                    if (k % N != 0) and (k != len(pts_with_theta) - 1):
                        continue

                    mi = int(round(pi))
                    mj = int(round(pj))
                    f.write(f"{mi}\t{mj}\t{th}\n")              

            # If we reach here, we successfully built a path → exit retry loop
            break

        else:
            print("Failed to find a path after maximum attempts.")


# bresenham algorithm for line generation on grid map
# from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
def bresenham(x1, y1, x2, y2):
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
