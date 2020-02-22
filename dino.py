from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from PIL import Image
from io import BytesIO
import skimage as skimage
from skimage import transform, color, exposure, io
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import numpy as np
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import argparse
from collections import deque
import random
import json


ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 1000. # timesteps to observe before training
EXPLORE = 10000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.0003



class Dino():
    def __init__(self):
        self.driver = webdriver.Chrome()

    def open(self):
        self.driver.get('chrome://dino')

    def up(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    
    def down(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def return_nor(self):
        pass

    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    
    def get_playing(self):
        return not self.driver.execute_script("return Runner.instance_.crashed")
    
    def restart(self):
        self.driver.execute_script("Runner.instance_.restart()")

    def get_score(self):
        score_digits = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_digits) 
        return int(score)

    def get_speed(self):
        try:
            speed_offset = self.driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].speedOffset")
        except:
            speed_offset = 0
        speed = self.driver.execute_script("return Runner.instance_.currentSpeed")
        speed = speed_offset + speed
        return speed

    def get_position(self):
        try:
            pos_obs = self.driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].xPos")
        except:
            pos_obs = 0
        dino_width = self.driver.execute_script("return Runner.instance_.tRex.config.WIDTH_DUCK")
        initial_distance  = self.driver.execute_script("return Runner.instance_.tRex.xPos")
        if pos_obs != 0:
            return pos_obs - (dino_width + initial_distance)
        else:
            return 1000             #If there is no obstacle, return 1000

    def get_size(self):
        try:
            return self.driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].width")
        except:
            return 0
    
    def pause(self):
        return self.driver.execute_script("return Runner.instance_.stop()")
    
    def resume(self):
        return self.driver.execute_script("return Runner.instance_.play()")

    def start(self):
        return self.up()

    def get_frame(self,count):
        canvas_details = self.driver.execute_script('return Runner.instance_.canvas.getBoundingClientRect()')
        frame = self.driver.get_screenshot_as_png()
        # frame_im = Image.open(BytesIO(frame))
        # frame_im = frame_im.crop((canvas_details['x'], canvas_details['y'], (canvas_details['x'] + canvas_details['width'])/2, canvas_details['y'] + canvas_details['height']))
        # frame_im = frame_im.resize((162,75))
        # fn = lambda x : 255 if x > 200 else 0
        frame_img = skimage.io.imread(BytesIO(frame))
        frame_img = skimage.color.rgb2gray(frame_img)
        frame_img = skimage.transform.resize(frame_img,(162,75))
        frame_img = skimage.exposure.rescale_intensity(frame_img,out_range=(0,255))
        frame_img = frame_img / 255.0
        # frame_im = frame_im.convert('L').point(fn, mode='1')
        # frame_im.save('/home/adubey/Desktop/SeleniumScripts/DinoAi/frames/frame-{}.png'.format(count))
        return frame_img

    def get_reward(self):
        if self.get_crashed()==True:
            return -1
        else:
            return 1

    def take_action(self,index):
        if index==1:
            self.up()
        elif index==2:
            self.down()
        else:
            self.return_nor()

    def auto_jump(self):
        count = 0
        while(True):
            count = count + 1
            self.save_frame(count)
            self.up()
            sleep(1)

class Model():
    def __init__(self):
        super().__init__()

    def buildmodel(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(162,75,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam',loss="mse", metrics=["accuracy"])

        # model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(162,75,3)))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(Flatten())
        # model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        # model.add(Activation('relu'))
        # model.add(Dense(3,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    
        # adam = Adam(lr=1e-6)
        # model.compile(loss='mse',optimizer=adam)
        print(model.metrics_names)
        return model

    def train(self,model,args):
        game = Dino()
        game.open()
        D = deque()
        frame = game.get_frame(1)
        stacked_frames = np.stack((frame,frame,frame),axis=2)
        stacked_frames = stacked_frames.reshape(1,stacked_frames.shape[0],stacked_frames.shape[1],stacked_frames.shape[2])
        print(stacked_frames.shape)
        if args['mode'] == 'Run':
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = FINAL_EPSILON
            print ("Now we load weight")
            model.load_weights("model.h5")
            adam = Adam(lr=LEARNING_RATE)
            model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")    
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        game.start()
        while(True):
            if game.get_crashed()==True:
                game.start()
            loss = 0
            Q_sa = 0
            targets = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([ACTIONS])
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    q = model.predict(stacked_frames)       #input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    a_t[max_Q] = 1
            
            #We reduce the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
            game.take_action(action_index)
            frame1 = game.get_frame(1)
            frame1 = frame1.reshape(1, frame1.shape[0], frame1.shape[1], 1)
            stacked_frames1 = np.append(frame1,stacked_frames[:,:,:,:2], axis=3)

            r_t = game.get_reward()

            # store the transition in D
            D.append((stacked_frames, action_index, r_t, stacked_frames1, game.get_crashed()))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
            
            #only train if done observing
            if t > OBSERVE:
                #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

                loss += model.train_on_batch(state_t, targets)[0]

            stacked_frames = stacked_frames1
            t = t+1

            #save progress every 500 iterations
            if t % 500 == 0:
                print("Save model")
                model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observing"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "exploring"
            else:
                state = "training"

            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX " , np.max(Q_sa), "/ Target ", np.max(targets), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")

    def playGame(self,args):
        model = self.buildmodel()
        self.train(model,args)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    game = Model()
    game.playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()


