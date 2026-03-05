import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        # action space is discrete with 5 actions, FORWARD, HARD_LEFT, HARD_RIGHT, LEFT, RIGHT
        self.action_space = spaces.Discrete(5)
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        
        self.last_x = None  # persistent state between callbacks where local variables die
        self.SLICE_HEIGHT = 80
        self.DARKNESS_THRESHOLD = 20
        self.FORWARD_SPEED = 0.4 # m/s
        self.TURN_SPEED = 0.6 # rad/s
        self.HARD_TURN_SPEED = 1.2 # rad/s

        self.FORWARD_REWARD = 5
        self.TURN_REWARD = 3
        self.HARD_TURN_REWARD = 1
        self.PENALTY = -200

    def process_image(self, data, action=None):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS
            @param action : The action taken in the previous timestep

            @returns (state, done)
        '''
        try:
            bgr8_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        NUM_BINS = 5
        state = [0, 0, 0, 0, 0] # default state with no line detected is all zeros
        done = False

        # extract region of interest (roi)
        cap_height, cap_width, _ = bgr8_image.shape
        if self.last_x is None: # if line never seen
            self.last_x = cap_width // 2
        y_start = cap_height - self.SLICE_HEIGHT
        roi = bgr8_image[y_start:cap_height, :]
        hsv_roi, avg_v = self.preprocess_slice(roi)
        
        # get line mask
        mask = self.get_line_mask(hsv_roi, avg_v)
        

        # update state
        current_x = self.find_cent_x_from_mask(mask)
        if current_x is not None:
            self.last_x = current_x
        else: # if line not detected, increment timeout
            self.timeout += 1
        if self.timeout > 4: # if line not detected for 5 consecutive frames, end episode
            done = True
            self.timeout = 0 # reset timeout for next episode

        state = self.get_state_from_x(self.last_x, cap_width, NUM_BINS)

        bin_width = cap_width // NUM_BINS

        # --- NEW: VISUALIZATION OVERLAY ---
        for i, val in enumerate(state):
            # Calculate the boundaries of this specific bin
            x_left = i * bin_width
            x_right = (i + 1) * bin_width
            x_center = x_left + (bin_width // 2)
            
            # Draw Bin Separators (Vertical Lines)
            cv2.line(bgr8_image, (x_left, 0), (x_left, cap_height), (255, 255, 255), 1)

            # Green for 1 (Active), Red/Gray for 0 (Inactive)
            color = (0, 255, 0) if val == 1 else (0, 0, 255)
            
            # Center the "0" or "1" text in the bin, slightly above the ROI slice
            text = str(val)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Get text size to perfectly center it
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x_center - (text_w // 2)
            text_y = y_start - 20 # 20 pixels above the slice
            
            cv2.putText(bgr8_image, text, (text_x, text_y), font, font_scale, color, thickness)

            if action is not None:
                # Map the numbers to human-readable strings
                action_names = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "HARD LEFT", 4: "HARD RIGHT"}
                command_text = action_names.get(action, "UNKNOWN")
                
                cv2.putText(bgr8_image, f"CMD: {command_text}", (10, 10 + text_h), 
                            font, font_scale, (255, 0, 0), thickness)

        # Draw the Centroid (Red Dot) for comparison
        if current_x is not None:
            cv2.circle(bgr8_image, (current_x, y_start + (self.SLICE_HEIGHT // 2)), 10, (0, 255, 255), -1)
        
        cv2.imshow("Line Follow View", bgr8_image)
        cv2.waitKey(1)

        return state, done

    def preprocess_slice(self, roi):
        """
            @brief Converts BGR to HSV and returns V channel average brightness.
            @param slice_image: Image in BGR format
            
            @returns HSV image and average brightness of V channel
        """
        hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv_image[:, :, 2])
        return hsv_image, avg_brightness
    
    def get_line_mask(self, hsv_slice, avg_v):
        """
            @brief Creates a binary mask for the line in the HSV slice. 

            @param hsv_slice: Image in HSV format
            @param avg_v: Average V channel (brightness) value 
            @return Binary mask image
        """
        upper_v = int(np.clip(avg_v - self.DARKNESS_THRESHOLD, 0, 255))
        lower_bound = np.array([0, 0, 0], dtype=np.uint8)
        upper_bound = np.array([179, 255, upper_v], dtype=np.uint8)
        return cv2.inRange(hsv_slice, lower_bound, upper_bound)

    def find_cent_x_from_mask(self, mask):
        """
            @brief Finds the centroid x-coordinate from a binary mask.
            @param mask: Binary image mask
            @return: x-coordinate of the centroid or None if not found
        """
        moment = cv2.moments(mask)
        # check if mask has white pixels to avoid division by zero
        if moment["m00"] != 0:
            cent_x = int(moment["m10"] / moment["m00"])
            return cent_x
        else:
            return None
    
    def get_state_from_x(self, x, width, num_bins):
        """
            @brief Converts an x-coordinate into a state representation based on bins.
            @param x: x-coordinate of the line centroid
            @param width: width of the image
            @param num_bins: number of bins to divide the image into
            
            @returns A one-hot encoded state array indicating which bin the line is in.
        """
        bin_size = width / num_bins
        # rounds down to nearest int, so bin_index is in [0, num_bins-1]
        bin_index = min(int(x // bin_size), num_bins - 1)
        state = [0] * num_bins
        if 0 <= bin_index < num_bins:
            state[bin_index] = 1
        return state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = self.FORWARD_SPEED
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.1 * self.FORWARD_SPEED
            vel_cmd.angular.z = self.TURN_SPEED
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.1 * self.FORWARD_SPEED
            vel_cmd.angular.z = -self.TURN_SPEED
        elif action == 3:  # HARD LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = self.HARD_TURN_SPEED
        elif action == 4:  # HARD RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -self.HARD_TURN_SPEED

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data, action)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = self.FORWARD_REWARD
            elif action == 1 or action == 2:  # LEFT or RIGHT
                reward = self.TURN_REWARD
            else:  # HARD LEFT or HARD RIGHT
                reward = self.HARD_TURN_REWARD
        else:
            reward = self.PENALTY

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
