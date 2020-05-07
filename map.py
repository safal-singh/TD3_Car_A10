# Self Driving Car

import cv2
import matplotlib.pyplot as plt
# Importing the libraries
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
# Importing the Kivy packages
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Line
from kivy.properties import BoundedNumericProperty, ReferenceListProperty, ObjectProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.vector import Vector

from ai import TD3, ReplayBuffer

# TRY USING RGBA FOR SHOWING TRANSPARENCY...!!!!

# Importing the Dqn object from our AI in ai.py

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
counter = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
policy = TD3(state_dim=22, action_dim=2, max_action=np.asarray([10., 2.]))
replay_buffer = ReplayBuffer()
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
max_velocity = 6.
min_velocity = 0
max_angle = +5.
# min_angle = -5
max_stuck = 100
stuck_count = 0

# Initializing the map
first_update = True


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global img
    global obs

    sand = np.zeros((longueur, largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img) / 255
    # car_map = img = PILImage.open("./images/MASK1.png")
    # plt.imshow(car_map)
    # plt.show()
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0


# Creating the car class

class Car(Widget):
    angle = BoundedNumericProperty(0)
    rotation = BoundedNumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        print('rotation: ', rotation)
        print('car velocity: ', self.velocity)
        print('car angle: ', self.angle)


# Creating the game class


class Game(Widget):
    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center  # 50, 50???
        self.car.velocity = Vector(2.0, 0)

    def evaluate_policy(self, learned_policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset()
            done = False
            while not done:
                action = learned_policy.select_action(np.array(obs))
                obs, reward, done, _ = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % avg_reward)
        print("---------------------------------------")
        return avg_reward

    def reset(self, done=False):
        global goal_x
        global goal_y
        # returns state/observation variables - snapshot, angle wrt target, distance from target
        if done:
            #   initialize agent with any of the starting points, chosen randomly
            starting_pts = [(1164, 614), (575, 535), (241, 559), (134, 280), (710, 227), (1155, 256)]
            index = np.random.randint(6)
            self.car.x = starting_pts[index][0]
            self.car.y = starting_pts[index][1]
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        # car_angle = Vector(*self.car.velocity).angle((6.0, 0.0))  # 180 for conversion to radians
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.

        contour = PILImage.open("./images/MASK1.png").convert("RGB")
        # rotate arrow head with car angle
        # car_arrow = car_arrow.resize((20, 10)).rotate(self.car.angle)
        # contour.paste(car_arrow, (int(self.car.x), largeur - int(self.car.y)))
        # contour = np.asarray(contour) / 255

        # colorTable = [255] * 256
        # colorTable[0] = 255  # anything black (0) will be made transparent
        #
        # mask = car_arrow.point(colorTable, '1')  # make the transparency mask
        #
        # # paste the overlay into the base image in the boundingBox using mask as a filter
        # contour.paste(car_arrow, (int(self.car.x), largeur - int(self.car.y)), car_arrow)
        # contour = np.asarray(contour) / 255

        draw = ImageDraw.Draw(contour)
        degree_to_radian = np.pi / 180.

        draw.polygon([(self.car.x + 20. * np.cos(degree_to_radian * self.car.angle),
                       largeur - (self.car.y + 20. * np.sin(degree_to_radian * self.car.angle))),
                      (self.car.x - 5. * np.sin(degree_to_radian * self.car.angle),
                       largeur - (self.car.y + 5. * np.cos(degree_to_radian * self.car.angle))),
                      (self.car.x + 5. * np.sin(degree_to_radian * self.car.angle),
                       largeur - (self.car.y - 5. * np.cos(degree_to_radian * self.car.angle)))], fill=(255, 0, 0))

        width = 100
        contour = np.asarray(contour)
        # snapshot = contour[(largeur - car_y) - width:(largeur - car_y) + width, car_x - width: car_x + width]
        # snapshot = snapshot[::2, ::2]
        # plt.imshow(snapshot)
        # plt.show()
        # snapshot = PILImage.fromarray(np.uint8(cm.gist_earth(snapshot)*255))

        # ROTATION IMAGE BY CAR ANGLE AND CROPPING THE CAR POSITION OUT OF IT
        # contour = PILImage.open("./images/MASK1.png").convert('L')
        # contour = np.asarray(contour) / 255
        # print('car position: ', self.car.pos)
        shape = (contour.shape[1], contour.shape[0])  # cv2.warpAffine expects shape in (length, height)
        matrix = cv2.getRotationMatrix2D(center=(self.car.x, self.car.y), angle=-self.car.angle, scale=1)
        image = cv2.warpAffine(src=contour, M=matrix, dsize=shape)
        car_x = int(self.car.x)
        car_y = int(self.car.y)

        # x = int(center[0] - width / 2)
        # y = int(center[1] - height / 2)
        #
        # image = image[y:y + height, x:x + width]
        # width = 100
        snapshot = image[(largeur - car_y) - width:(largeur - car_y) + width, car_x - width: car_x + width]
        # snapshot[50, 50] = (0, 0, 0)
        # plt.imshow(snapshot)
        # plt.show()

        return snapshot, orientation, distance

    def select_random_action(self):
        action = np.random.rand(1, 2)
        action = np.multiply(action, np.asarray([2*max_angle, max_velocity]))
        action.resize(2)

        action[0] -= max_angle
        # action[1] -= max_velocity
        return action

    def step(self, action):
        global last_distance
        global stuck_count
        print('stuck count: ', stuck_count)
        # action - angle, velocity
        # observation - image, orientation wrt target, distance from target
        self.car.move(action[0])
        self.car.velocity = Vector(float(action[1]), 0).rotate(self.car.angle)

        reward = 0
        new_obs = self.reset()
        done = False
        distance = new_obs[2]
        if distance < 25:
            print('reached target!!!')
            reward += 5
            done = True
        elif distance < last_distance:
            print('closer to target')
            reward += 0.1
        else:
            print('away from target')
            reward += -0.2

        if stuck_count > max_stuck:
            print('stuck in the boundary for too long!')
            reward += -2
            done = True

        if self.car.x < 10:
            self.car.x = 10
            stuck_count += 1
            reward += -1
        elif self.car.x > self.width - 10:
            self.car.x = self.width - 10
            stuck_count += 1
            reward += -1
        elif self.car.y < 10:
            self.car.y = 10
            stuck_count += 1
            reward += -1
        elif self.car.y > self.height - 10:
            self.car.y = self.height - 10
            stuck_count += 1
            reward += -1
        else:
            stuck_count = 0

        if sand[int(self.car.x), int(self.car.y)] > 0:
            reward += -1
        else:  # negative reward for spending more time reaching the goal
            reward += -0.2

        print('car position: ', self.car.pos)

        print(goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
              im.read_pixel(int(self.car.x), int(self.car.y)))
        last_distance = distance

        return new_obs, reward, done, False

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global counter

        longueur = self.width
        largeur = self.height
        global obs
        global done
        global total_timesteps
        global start_timesteps
        global batch_size
        global episode_num
        global episode_reward
        global episode_timesteps
        global discount
        global tau
        global policy_noise
        global noise_clip
        global policy_freq
        global timesteps_since_eval
        global eval_freq
        global evaluations
        global file_name
        global expl_noise
        global max_episode_steps

        if first_update:
            init()
            self.car.center = (708, 226)
            seed = 0  # Random seed number
            start_timesteps = 1e3  # Number of iterations/timesteps before which the model randomly chooses an action,
            # and after which it starts to use the policy network
            eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
            # max_timesteps = 5e5  # Total number of iterations/timesteps
            save_models = True  # Boolean checker whether or not to save the pre-trained model
            expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
            batch_size = 100  # Size of the batch
            discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
            tau = 0.005  # Target network update rate
            policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
            noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
            policy_freq = 2  # Number of iterations to wait before the policy network (Actor model) is updated
            total_timesteps = 0
            done = False
            episode_num = 0
            timesteps_since_eval = 0
            episode_reward = 0
            episode_timesteps = 0
            evaluations = list()
            file_name = 'TD3'
            max_episode_steps = 1e4
            obs = self.reset()

        if done:
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0 and total_timesteps > batch_size:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                              episode_reward))
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                             policy_freq)

            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(self.evaluate_policy(learned_policy=policy))
                policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % file_name, evaluations)

            # When the training step is done, we reset the state of the environment
            # define reset method to return state vars
            obs = self.reset(done=True)

            # Set the Done to False
            done = False

            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Before 10000 timesteps, we play random actions
        # action space consists of direction and velocity
        if total_timesteps < start_timesteps:
            action = self.select_random_action()
        else:  # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=2)).clip(np.asarray([-max_angle, min_velocity]),
                                                                                 np.asarray([max_angle, max_velocity]))

            # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = self.step(action)

        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)

        # We increase the total reward
        episode_reward += reward

        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of
        # the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

        # We add the last policy evaluation to our list of evaluations and we save our model
        # evaluations.append(evaluate_policy(policy))
        # if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
        # np.save("./results/%s" % (file_name), evaluations)


# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8") * 255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 1

            last_x = x
            last_y = y


# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
