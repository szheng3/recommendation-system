# Deep Reinforcement Learned Recommenders using Item Features
## Content
- [Item Features](#item-features)
- [Running the Code](#instructions-for-running-the-code)
- [RetailRocket Results](#retailrocket-results)
- [H&M Results](#hm-results)
- [Conclusion](#notes)
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
| SASRec-SNQN with item features(lambda=0)   |0.000010|0.000036|0.000065|0.000077|0.000025|0.000110|0.000220|0.000270
| SASRec-SNQN with item features(lambda=0.5) |0.007478|0.009129|0.010264|0.011161|0.010963|0.016035|0.020337|0.024132
| SASRec-SNQN with item features(lambda=1.0) |0.134778|0.144440|0.148851|0.151738|0.170439|0.200320|0.216980|0.229194
| SASRec-SNQN without item features          |0.134854|0.144680|0.149209|0.152096|0.170372|0.200725|0.217825|0.230039

</td></tr> 
</table>

<table>
<tr><th> Purchase </th>
<tr><td>

|                                            |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|--------------------------------------------|------|-------|-------|-------|----|-----|-----|-----|
| SASRec-SNQN with item features(lambda=0)   |0.000000|0.000000|0.000053|0.000186|0.000000|0.000000|0.000189|0.000756
| SASRec-SNQN with item features(lambda=0.5) |0.018287|0.020960|0.022762|0.024020|0.026838|0.034965|0.041769|0.047061
| SASRec-SNQN with item features(lambda=1.0) |0.311309|0.322806|0.328161|0.331062|0.363636|0.398979|0.419202|0.431487
| SASRec-SNQN without item features          |0.319153|0.329729|0.335778|0.339388|0.371763|0.404649|0.427518|0.442827

</td></tr> 
</table>

### H&M Results

in 4000th batch

<table>
<tr><th> Purchase </th>
<tr><td>


|                                            |NDCG@5|NDCG@10|NDCG@15|NDCG@20|HR@5|HR@10|HR@15|HR@20|
|--------------------------------------------|------|-------|-------|-------|----|-----|-----|-----|
| SASRec-SNQN with item features(lambda=0)   |0.000365|0.000533|0.000559|0.000570|0.000552|0.001056|0.001152|0.001200
| SASRec-SNQN with item features(lambda=0.5) |0.000184|0.000240|0.000246|0.000252|0.000312|0.000480|0.000504|0.000528
| SASRec-SNQN with item features(lambda=1.0) |0.023141|0.025860|0.027247|0.028268|0.031075|0.039498|0.044753|0.049097
| SASRec-SNQN without item features          |0.021394|0.024258|0.025746|0.026787|0.028340|0.037218|0.042881|0.047297

</td></tr> </table>

### Conclusion

According to the table above, increasing the lambda value leads to better results. This suggests that combining item features with SNQN is effective. However, when lambda is set to 0 (i.e., using item features without SNQN), it has a negligible impact on the result. When lambda is set to 0.5 (i.e., using item features and SNQN equally), the performance is worse than when lambda is set to 1 (i.e., using SNQN without item features). This implies that, in general, item features help when there is a cold start (i.e., when there is little user history available), whereas SNQN becomes more important when there is sufficient user history.






### Notes
Please refer to [README.md](https://github.com/szheng3/recommendation-system/blob/main/README.md) for details on evaluation metrics.