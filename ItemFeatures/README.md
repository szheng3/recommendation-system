# Deep Reinforcement Learned Recommenders using Item Features
## Content
- [Item Features](#item-features)
- [Running the Code](#instructions-for-running-the-code)
- [RetailRocket Results](#retailrocket-results)
- [H&M Results](#hm-results)
- [Notes](#notes)

## Item Features
We added item features mentioned in the HRNN paper with below formula:

<img width="625" alt="CleanShot 2023-05-02 at 23 45 00@2x" src="https://user-images.githubusercontent.com/16725501/235829137-a75a658c-95e1-4d54-9117-feeb746d17c9.png">

code is implemented below located in the SNQN_item_feature.py file
```python
self.item_features = tf.compat.v1.placeholder(tf.float32, [None, item_num, self.feature_dim])
self.lambda_values = tf.compat.v1.placeholder(tf.float32, [None])

self.lambda_values_expanded = tf.expand_dims(self.lambda_values, axis=-1)

self.feature_embedding = tf.compat.v1.layers.dense(self.item_features, self.hidden_size + 1,
                                                   activation=None)
dot_product = tf.matmul(self.states_hidden,
                        tf.transpose(self.feature_embedding[:, :, :-1], perm=[0, 2, 1]))
reshaped_dot_product = tf.reshape(dot_product, shape=(-1, item_num))

self.phi_prime = reshaped_dot_product + self.feature_embedding[:, :, -1]

self.final_score = tf.add(
    tf.multiply(self.lambda_values_expanded, self.output2),
    tf.multiply(tf.subtract(1.0, self.lambda_values_expanded), self.phi_prime)
                )
```
## Instructions for Running the Code
**Train SASRec-SNQN on RetailRocket:**

Open `/DRL2/SA2C_Recommender.ipynb` in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

**Train SASRec-SNQN with item features on RetailRocket:**

Open `/DRL2/SNQN_Recommender.ipynb` in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

**Train SASRec-SA2C on HM:**

Open `/DRL2/HM_SA2C_Recommender.ipynb` in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

**Train SASRec-SNQN on HM:**

Open `/DRL2/HM_SNQN_Recommender.ipynb` in Google Colab. This notebook contains all code necessary to run training and view results. This notebook will run both with and without CQL Loss. Evaluation Metrics can be found below.

### RetailRocket Results

<table>
<tr><th> Clicks </th>
<tr><td>

|    |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|----|------|-------|-------|-------|----|-----|-----|-----|
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

### H&M Results

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

### Notes
Please refer to [README.md](https://github.com/szheng3/recommendation-system/blob/main/README.md) for details on evaluation metrics.