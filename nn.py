# record: model converged on trial 20, after which diverges
# and then converged to another solution and diverges again

from game import controlled_run

import numpy as np
from game import DO_NOTHING
from game import JUMP

games_count = 0
total_number_of_games = 30

import tensorflow
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# model creation
model = Sequential()
model.add(Dense(input_dim=1, activation='sigmoid', init='uniform', output_dim=1))
model.add(Dense(activation='sigmoid', output_dim=1, init='uniform'))
model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

x_train = np.array([])
y_train = np.array([])

class Wrapper(object):
    last_value = -1
    jumped = False

    def __init__(self):
        controlled_run(self, 0)

    def control(self, values):
        global x_train
        global y_train
        global val

        print(values)

        # assigning values
        action = values['action']
        old_val = values['old_closest_enemy']/1000
        val = values['closest_enemy']

        # if no enemies, do nothing
        if val == -1:
            return DO_NOTHING

        # creating data points from good and bad jumps
        # if we detect enemy
        else:
            # if we made a prediction to jump AND do not score
            if action == 1:
                if values['score_increased'] == 0:
                    x_train = np.append(x_train, np.array(old_val))
                    y_train = np.append(y_train, [0])
            # if we made a jump and scored
                else:
                    x_train = np.append(x_train, np.array(old_val))
                    y_train = np.append(y_train, [1])

        # make a prediction, starts with random and eventually converges
        prediction = model.predict_classes(np.array([values['closest_enemy']]) / 1000)
        self.last_value = val/1000

        print("predictions:")
        print(prediction)

        if prediction >= 0.5:
            self.jumped = True
            return JUMP
        else:
            self.jumped = False
            return DO_NOTHING

    def gameover(self, score):
        global games_count
        global x_train
        global y_train

        # when game ends, append a data point based on how we lost
        if games_count >= 0:
            x_train = np.append(x_train, self.last_value)
            if self.jumped:
                y_train = np.append(y_train, [0])
            else:
                y_train = np.append(y_train, [1])

        # training the model at the end of each run
            model.fit(x_train, y_train, epochs=10, verbose=1, shuffle=1)
            print(y_train)

        games_count += 1

        print("x_train:")
        print(x_train)
        print("y_train ")
        print(y_train)

        if games_count >= total_number_of_games:
            return
        controlled_run(self, games_count)


w = Wrapper()

