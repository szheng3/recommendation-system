# Deep Reinforcement Learned Recommenders with CQL
## Content
- [Conservative Q-Learning](#conservative-q-learning)
- [Running the Code](#instructions-for-running-the-code)
- [RetailRocket Results](#retailrocket-results)
- [H&M Results](#hm-results)
- [Notes](#notes)

## Conservative Q-Learning
Conservative Q-Learning (CQL) algorithm is a SAC-based data-driven deep reinforcement learning algorithm, which achieves state-of-the-art performance in offline RL problems. CQL mitigates overestimation error by minimizing action-values under the current policy and maximizing values under the data distribution, enforcing a conservative policy update by incorporating a penalty term based on the estimation of the expected maximum action value under the current policy. As a result, the CQL loss encourages the agent to explore more efficiently by contrasting its Q-values against the Q-values of other actions that are not in the dataset. Discrete CQL, as implemented in d3rlpy, applies the same principles to discrete action spaces, making it suitable for use with algorithms like DQN.

This loss function is given by:

```
CQL_loss = E(s, a) [log (1 + exp(Q(s, a) - Q(s, a') + margin))]
```
where E(s, a) denotes expectation over states and actions, Q(s, a) is the Q-value for the state-action pair, and Q(s, a') is the Q-value for the state and other actions. 

## Instructions for Running the Code

**Train SASRec-SA2C on RetailRocket:**

Open 'DLR2/SA2C_Recommender.ipynb' in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

**Train SASRec-SNQN on RetailRocket:**

Open 'DLR2/SNQN_Recommender.ipynb' in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

**Train SASRec-SA2C on HM:**

Open 'DLR2/HM_SA2C_Recommender.ipynb' in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

**Train SASRec-SNQN on HM:**

Open 'DLR2/HM_SNQN_Recommender.ipynb' in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

### RetailRocket Results

#### Without CQL Loss

<table>
<tr><th> Clicks </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|-------|----|-----|-----|-----|
|SASRec-SA2C|0.2229|0.2395|0.2468|0.2513|0.2833|0.3344|0.3622|0.3812
|SASRec-SNQN|0.2178|0.2366|0.2447|0.2497|0.2839|0.3415|0.3722|0.3931
</td></tr> </table>

<table>
<tr><th> Purchase </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C|0.5129|0.5264|0.5316|0.5348|0.5903|0.6318|0.6514|0.6648
|SASRec-SNQN|0.4717|0.4909|0.4981|0.5020|0.5710|0.6302|0.6572|0.6741
</td></tr> </table>

#### With CQL Loss 
(For SASRec-SA2C, alpha=1.0)

<table>
<tr><th> Clicks </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NADCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C|0.2263|0.2428|0.2501|0.2547|0.2879|0.3386|0.3663|0.3857
|SASRec-SNQN||||||||
</td></tr> </table>

<table>
<tr><th> Purchase </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NADCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C|0.5153|0.5291|0.5346|0.5376|0.5961|0.6384|0.6593|0.6721
|SASRec-SNQN||||||||
</td></tr> </table>

### H&M Results

#### Without CQL Loss

<table>
<tr><th> Clicks </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|--------|----|-----|-----|-----|
|SASRec-SA2C||||||||
|SASRec-SNQN||||||||
</td></tr> </table>

<table>
<tr><th> Purchase </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
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

### Notes
Please refer to [README.md] (https://github.com/szheng3/recommendation-system/blob/main/README.md) for details on evaluation metrics.