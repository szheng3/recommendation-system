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

### Retail Rocket

### Dataset 2

## Instructions for Running the Code

## Evaluation

### Metrics

**Normalized Discounted Cumulative Gain (NDCG)**: NDCG is an offline reinforcement learning evaluation metric that takes two factors into consideration:
    - how relevant are the results?
    - how accurate is the ordering of these results?.

To understand how NDCG works, we need to start with an understanding of what *gain* is. Gain is simply defined as the relevance score for a given recommendation. Depending on your use case, there are different methods to measure the relevance of a recommendation. To find the *cumulative gain*, we can simply calculate the sum of gains across all the recommendations. Usually this is calculated for the first K recommendations.

$$ CG = \sum^{K}_{i=1}G_i $$

Now that we have calculated the relevancy of the recommendations, we want to understand how good the ordering of the recommendations is. This is where *discounted cumulative gain* comes in. discounted cumulative gain weighs the relevance of recommendations by the item's position in the reocmmendations. The top recommended items get the highest weight.

$$ DCG = \sum^{K}_{i=1} \frac{G_i}{log_2(i+1)}$$

Finally we can calculate NDCG by normalizing the dicounted cumulative gain. To do this, we will divide by the *ideal discounted cumulative gain*:

$$ IDCG = \sum^{K}_{i=1} \frac{G^{ideal}_i}{log_2(i+1)}$$
$$ NDCG = $\frac{DCG}{IDCG}$$

A score of 0 means the recommendations are useless, while a score of 1 means they are perfect. If you would like to learn more, here is a great [resource](https://machinelearninginterview.com/topics/machine-learning/ndcg-evaluation-metric-for-recommender-systems/).

**Hit Ratio (HR)**: Hit ratio represents the fraction of users for which the "correct answer" is included in the top K items of their recommendation list. Hit ratio has a value between 0 and 1, where 0 means no customers have the best item in their recommendation list, and 1 means that every customer has the best item included in the first K items of their recommendation list.

If we define *C* to be the number of users with the correct answer recommended to them, and *T* to tbe the total number of users, we can calculate hit ratio with the following equation:

$$ HR = \frac{C}{T}$$

### Results

#### Without CQL Loss

<table>
<tr><th> Clicks </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NADCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C|0.5129|0.5265|0.5316|0.5348|0.5903|0.6318|0.6514|0.6648
|SASRec-SNQN||||||||
</td></tr> </table>

<table>
<tr><th> Purchase </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NADCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C||||||||
|SASRec-SNQN||||||||
</td></tr> </table>

#### With CQL Loss

<table>
<tr><th> Clicks </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NADCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C||||||||
|SASRec-SNQN||||||||
</td></tr> </table>

<table>
<tr><th> Purchase </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NADCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C||||||||
|SASRec-SNQN||||||||
</td></tr> </table>

## Contributions


## References
1. Kumar et al., Conservative Q-Learning for Offline Reinforcement Learning, arXiv Aug 2020. (https://arxiv.org/pdf/2006.04779.pdf)
2. Xin Xin et al.,Supervised Advantage Actor-Critic for Recommender Systems, arXiv Nov 2021. (https://arxiv.org/abs/2111.03474)
