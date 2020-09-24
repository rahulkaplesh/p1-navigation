# Project 1: Navigation

### Introduction

For this project, an agent is trained to collect bananas in a square world !!

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The excercise is given as two challenges 

1.) One version of the enevironment gave a vector state space using which the machine learning agent could be trained 

2.) Another version of the environment gave a RGB image as the state space which had to be run through a cnn to decipher the state space and take actions accordingly

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. For the visual agent , download a new Unity Environment. The steps are given below : 

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

### Instruction

1. You can start by cloning this repo as i have downloaded and included the unity versions of the environment which i used to do it as well as the code works for the directory structure

### Directory Structure

- Banana_Windows_x86_x64 : Directory containing the unity environment which gave the vector state space to work with.
- DoubleDQN : Implemetation of Double DQN to solve the environment which returns the vector state space .64 x 64 layer fc network was used as deep nn to estimate the action values.
- DobubleDQN_moreFCLayers : Implemetation of Double DQN to solve the environment which returns the vector state space .128 x 64 x 32 layer fc network was used as deep nn to estimate the action values.
- DoubleDQN4LayersExpReplay : Implementation of Double DQN with prioritised replay to solve the environment which returns the vector state space. 128 x 64 x 32 layer fc deep nn was used to estimate the action values.
- Dueling_NN_DQN : Implementation of Double DQN with duelling nueral network for estimating the value of the state. Both the networks were 128 x 64 x 32 layer fc nn to extimate the action values.
- DQN_RNN_PixelState : Implementation of Double DQN on the environment which returned the RGB image of what is seen by the agent . A set of 4 such images were stitched together to form a frame which was to run through a CNN . Any new state was added on to the end of the stack and the image from the top of the stack was removed. This frame was then run through a 3D CNN to extract the state 
features out of the frame of 4 state captures. These features were used to determine the actions which the agent can take within those states



### Result files 

This repository also consists of a Report.pdf file which contain some of my observations.
