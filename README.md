# Deep RL Recommender System for E-commerce --  Take Home Project for AIPI531

## Team Members


## Conservative Q-Learning
Conservative Q-Learning (CQL) algorithm is a SAC-based data-driven deep reinforcement learning algorithm, which achieves state-of-the-art performance in offline RL problems. CQL mitigates overestimation error by minimizing action-values under the current policy and maximizing values under data distribution for underestimation issue. Its goal is to improve the performance by incorporating an additional loss term in the training process. As a result, the CQL loss encourages the agent to explore more efficiently by contrasting its Q-values against the Q-values of other actions that are not in the dataset. 

This loss function is given by:
```
CQL_loss = E(s, a) [log (1 + exp(Q(s, a) - Q(s, a') + margin))]
```
where E(s, a) denotes expectation over states and actions, Q(s, a) is the Q-value for the state-action pair, and Q(s, a') is the Q-value for the state and other actions. The margin ...



## Datasets

## Instructions for Running the Code



## Contributions


## References
1. Kumar et al., Conservative Q-Learning for Offline Reinforcement Learning, arXiv Aug 2020. (https://arxiv.org/pdf/2006.04779.pdf)
2. Xin Xin et al.,Supervised Advantage Actor-Critic for Recommender Systems, arXiv Nov 2021. (https://arxiv.org/abs/2111.03474)
