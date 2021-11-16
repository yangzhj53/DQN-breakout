from typing import (
    Tuple,
)

import torch
import numpy as np

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)



class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            full_sink: bool = True,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        sink = lambda x: x.to(device) if full_sink else x
        self.__m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size


"""
code from
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
"""
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.size = 0
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def get_min(self):
        return min(self.tree[self.capacity - 1:self.capacity + self.size - 1])
    
    @property
    def total_p(self):
        return self.tree[0]  # the root


"""
code from
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py

modified to suit the original replay memory
"""
class PriorityReplayMemory(object):
    """
    The code I referenced also referenced another person's code:

    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.channels = channels
        self.tree = SumTree(capacity)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, [folded_state, action, reward, done]) 

    def sample(self, batch_size):
    
        b_states = torch.zeros((batch_size, self.channels, 84, 84), dtype=torch.uint8)
        b_actions = torch.zeros((batch_size, 1), dtype=torch.long)
        b_rewards = torch.zeros((batch_size, 1), dtype=torch.int8)
        b_next = torch.zeros((batch_size, self.channels, 84, 84), dtype=torch.uint8)
        b_dones = torch.zeros((batch_size, 1), dtype=torch.bool)
        b_treeNode = np.empty((batch_size,), dtype=np.uint64)
        ISWeights = torch.zeros((batch_size, 1))   
        
        pri_seg = self.tree.total_p / batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = self.tree.get_min() / self.tree.total_p     # for later calculate ISweight
        
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            
            b_states[i] = data[0][0][0:4]
            b_actions[i] = data[1]
            b_rewards[i] = data[2]
            b_next[i] = data[0][0][1:]
            b_dones[i] = data[3]
            b_treeNode[i] = idx
            try:
                ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            except RuntimeWarning:
                data = self.predata

            self.predata = data.copy()
            
        b_states = b_states.to(self.__device).float()
        b_next = b_next.to(self.__device).float()
        b_actions = b_actions.to(self.__device)
        b_rewards = b_rewards.to(self.__device).float()
        b_dones = b_dones.to(self.__device).float()
        ISWeights = ISWeights.to(self.__device).float()
        
        return b_states, b_actions, b_rewards, b_next, b_dones, b_treeNode, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    
    def __len__(self) -> int:
        return self.__size         
            