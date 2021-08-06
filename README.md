# Chess Engine

The project was created for the purpose of learning reinforcement learning
It is based on Giraffe Chess.
The paper: https://arxiv.org/pdf/1509.01549.pdf

## To run
Download the training data and unzip and move the folder into the main directory
Create the weights by running TDLeaf.py
Run GUI.py to play the game

### Training Data
Download Training Data (uncommented version) from
https://ccrl.chessdom.com/ccrl/4040/games.html , unzip and move to main folder

### Neural Network
The neural network uses temporal difference learning to improve it's own
evaluation function. The neural network uses mini batch to train due to 
hardware constraints for loading the full dataset into memory.

### Notes
Computer is currently on minimax mode with depth 1   
To change, go into Environment.py, init, line 33 and change algo   
To turn on self-play, go to Environment.py, run(), line 115 and enable self-play    
To enable dqn mode, run TDLeaf.py to create a weights file for the model   
Configurations for the training can be changed in test_TDLeaf()

### Results
The neural network was unable to learn basic strategies.    
This may be due to many factors including:    
- Lack of training; I only used 500 randomly selected batches out of the whole dataset.
This was due to hardware and time limitations. 500 batches took about 8 hours to complete.
- Hard to test; Since the training took a long time, it was hard to create multiple models
to see which hyper parameters and features could have affected the model.
- Some features include: 
    - Loss Function (Currently compares tanh, [-1,1], to total_loss [-100_000,100_000])
    - Input for neural network does not have bias
    - Learning rate may have been too small for training side
    - May require more data
    - Evaluation function may be too broad/ simple
 