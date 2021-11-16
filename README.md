# DQN-breakout
This is the mid-term assignment of 2019 Reinforcement Learning and Game Theory of Sun Yat sen University, and it is modified from https://gitee.com/goluke/dqn-breakout.<br>
Before uploading, we adjusted the file structure of the project. If you need to reproduce the experimental results, please adjust the file path according to the content.

## A quick preview of our work

<img src="pic\1.png" alt="pic1" style="zoom:100%;" />

**In this project, we finished these following requirements**

* Boost the training speed using Prioritized Experience Replay :white_check_mark:

* Improve the performance using Dueling DQN}\checkmark:white_check_mark:

* Stabilize the movement of the paddle:white_check_mark:

  *The model training is very successful and converges quickly*

**In addition, we have made innovations in other aspects**

* We added ddqn model and integrated ddqn model with dueling dqn model
* We explore the method of fast training model, which can quickly train convergence in a very short number of training times
* During the training, we found several interesting agent models. Their strategy is to save life rather than choose to pass the customs. Therefore, such models will never lose the game and continue the game forever while getting high scores.

## Stable version using state mechine（not the highest score)

| ![](video\breakout.gif) | <img src="pic\statemachine.drawio.png" alt="statemachine.drawio" style="zoom: 50%;" /> |
| ----------------------- | ------------------------------------------------------------ |



## More Experiments

### Basic implement of DQN

<img src="pic\pic1.png" alt="pic1" style="zoom:50%;" />

### Using Dueling-DQN

<img src="pic\pic3.png" alt="pic3" style="zoom:50%;" />

### Using Double-DQN

<img src="pic\pic2.png" alt="pic2" style="zoom:50%;" />

### Using DDQN+Duel

<img src="pic\pic4.png" alt="pic4" style="zoom:50%;" />

### Using DDQN+Duel+Prioritised Experience Replay

<img src="pic\pic6.png" alt="pic6" style="zoom:50%;" />

### Using DDQN+Duel with eps setting to 0(no greedy)

<img src="pic\pic5.png" alt="pic5" style="zoom:50%;" />

### Using DDQN+Duel+PR with state machine

<img src="pic\pic7.png" alt="pic7" style="zoom:50%;" />



