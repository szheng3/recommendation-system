# Deep Reinforcement Learned Recommenders using Item Features
## Content
- [Item Features](#item-features)
- [Running the Code](#instructions-for-running-the-code)
- [RetailRocket Results](#retailrocket-results)
- [H&M Results](#hm-results)
- [Notes](#notes)

## Item Features
In our project, we combine the SNQN models with item features mentioned in the HRNN paper using the following formula:
<p align="center">
<img src="https://user-images.githubusercontent.com/16725501/235830850-57f03f1c-9d39-4b96-b31e-c5f01e724b69.png" width="50%" height="50%" />
</p>

The implementation of this code can be found in the `SNQN_item_feature.py` file. We create a dense layer for the feature embedding and compute the dot product between the states hidden and the feature embedding:

```python
self.feature_embedding = tf.compat.v1.layers.dense(self.item_features, self.hidden_size + 1,
                                                   activation=None)
dot_product = tf.matmul(self.states_hidden,
                        tf.transpose(self.feature_embedding[:, :, :-1], perm=[0, 2, 1]))

```
We then reshape the dot product and add it to the bias term in the feature embedding to obtain phi_prime:
```python
reshaped_dot_product = tf.reshape(dot_product, shape=(-1, item_num))
self.phi_prime = reshaped_dot_product + self.feature_embedding[:, :, -1]

```
Finally, we compute the final score using the lambda values, the output2 value, and phi_prime:
```python
self.final_score = tf.add(
    tf.multiply(self.lambda_values_expanded, self.output2),
    tf.multiply(tf.subtract(1.0, self.lambda_values_expanded), self.phi_prime)
)
```
Overall, this implementation allows us to combine the strengths of SNQN models and item features.


## Instructions for Running the Code

**Train SASRec-SNQN with item features on RetailRocket:**

Open `RC_SNQN_ITEM_FEATURE.ipynb` in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without item features. Evaluation Metrics can be found below.


**Train SASRec-SNQN on HM:**

Open `HM_SNQN_ITEM_FEATURE.ipynb` in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without item features. Evaluation Metrics can be found below.

### RetailRocket Results

in 4000th batch

<table>
<tr><th> Clicks </th>
<tr><td>

|                                            |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|--------------------------------------------|------|-------|-------|-------|----|-----|-----|-----|
| SASRec-SNQN with item features(lambda=0)   ||||||||
| SASRec-SNQN with item features(lambda=0.5) ||||||||
| SASRec-SNQN with item features(lambda=1.0) ||||||||
| SASRec-SNQN without item features          ||||||||

</td></tr> 
</table>

<table>
<tr><th> Purchase </th>
<tr><td>

|                                            |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|--------------------------------------------|------|-------|-------|-------|----|-----|-----|-----|
| SASRec-SNQN with item features(lambda=0)   ||||||||
| SASRec-SNQN with item features(lambda=0.5) ||||||||
| SASRec-SNQN with item features(lambda=1.0) ||||||||
| SASRec-SNQN without item features          ||||||||
</td></tr> 
</table>

### H&M Results

in 4000th batch

<table>
<tr><th> Purchase </th>
<tr><td>


|                                            |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|--------------------------------------------|------|-------|-------|-------|----|-----|-----|-----|
| SASRec-SNQN with item features(lambda=0)   ||||||||
| SASRec-SNQN with item features(lambda=0.5) ||||||||
| SASRec-SNQN with item features(lambda=1.0) ||||||||
| SASRec-SNQN without item features          |0.021093|0.023655|0.025066|0.026047|0.028052|0.036019|0.036019|

</td></tr> </table>

### Notes
Please refer to [README.md](https://github.com/szheng3/recommendation-system/blob/main/README.md) for details on evaluation metrics.