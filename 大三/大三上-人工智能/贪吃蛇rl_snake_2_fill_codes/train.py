#!/usr/bin/env python3

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser
from keras.models import load_model


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """
    model = Sequential()
    ################################# Fill in codes #################################
    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size = (3,3),
        strides = (1,1),
        data_format = 'channels_first',
        input_shape = (num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kerenl_size=(3,3),
        strides = (1,1),
        data_format = 'channels_first'
    )
    )
    model.add(Activation('relu'))

    # hint: use 2 convolution layers with relu activations attached.
    # hint: use 2d convolution with 3x3 kernal_size and 1x1 strides.              
    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env,num_actions))
    # hint: use 3 dense layers and relu activations attached to the 2nd dense layer.          
              
    

                    
    ################################# End filling #################################
    
    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


def main():
    env = create_snake_environment(r'10x10-blank.json')
    # model = create_dqn_model(env, num_last_frames=4)
    model = load_model(r'dqn-final.model')

    agent = DeepQNetworkAgent(
        model=model,
        memory_size=-1,
        num_last_frames=model.input_shape[1]
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=3000,
        discount_factor=0.95
    )

if __name__ == '__main__':
    main()
