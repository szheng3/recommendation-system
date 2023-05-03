# Deep Reinforcement Learning Recommender System for E-commerce --  Take Home Project for AIPI531

> #### Team Members: Chad Miller, Andrew Bonafede, Shuai Zheng, Yilun Wu, Bryce Whitney

## Content
- [About the Project](#overview)
- [Dataset Descriptions](#datasets)
- [Evaluation Metrics](#evaluation)
- [Contributions](#contributions)
- [References](#references)

## Overview

Objectives:

1.Train different session (contextual, sequential) based product recommendation recommenders
for E-commerce use cases and compare the performances of the recommenders.
2.Include the CQL loss to improve the model performance.
3.Include item and/or user features as side information for cold items/users.

Requirements:

In the deliverables and experiments, one of the recommenders needs to be a Deep RL
recommender [DRL2] and at least two different datasets are used for training/testing. Also, at
least two offline evaluation metrics are used for benchmarking

### CQL
Please see [README.md](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/README.md) for instructions on how to run our CQL Loss trained recommenders. The associated code can be found [here](https://github.com/szheng3/recommendation-system/tree/main/Explore_CQL)

### Item Features
Please see [README.md](https://github.com/szheng3/recommendation-system/blob/main/ItemFeatures/README.md) for instructions on how to run our Item Features trained recommenders. The associated code can be found [here](https://github.com/szheng3/recommendation-system/tree/main/ItemFeatures)

## Datasets

### RetailRocket

The [RetailRocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) contains real world ecommerce data from [RetailRocket](https://retailrocket.net/). The data contains a couple important files. `events.csv` contains data on customer behavior such as when they viewed items, added items to their cart, purchased items, etc. `item_properties.csv`contains properties specific to each item. The data represents a 4.5 month span and contains over 2.75 million events from over 1.4 million unique visitors to the website.

#### EDA
See [`EDA_RetailRocket.ipynb`](EDA_RetailRocket.ipynb)

### H&M

The [H&M Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=transactions_train.csv) contains real-world purchase history of customers from [H&M](https://www2.hm.com/en_us/index.html). For our purposes, the most import file is `transactions_train.csv` which contains the purchases of each customer. This includes which customer made the purchase, when they made the purchase, and what item they purchased.

#### EDA
See [`EDA_HM.ipynb`](EDA_HM.ipynb)

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
$$ NDCG = \frac{DCG}{IDCG}$$

A score of 0 means the recommendations are useless, while a score of 1 means they are perfect. If you would like to learn more, here is a great [resource](https://machinelearninginterview.com/topics/machine-learning/ndcg-evaluation-metric-for-recommender-systems/).

**Hit Ratio (HR)**: Hit ratio represents the fraction of users for which the "correct answer" is included in the top K items of their recommendation list. Hit ratio has a value between 0 and 1, where 0 means no customers have the best item in their recommendation list, and 1 means that every customer has the best item included in the first K items of their recommendation list.

If we define *C* to be the number of users with the correct answer recommended to them, and *T* to tbe the total number of users, we can calculate hit ratio with the following equation:

$$ HR = \frac{C}{T}$$

## Contributions

Chad Miller:

Shuai Zheng: worked on the item features of our project, combining SNQN with item features in the `SNQN_item_features.py` file for both RetailRocket and HM datasets, and creating a version of SNQN without item features in the `SNQN_new.py` file. He also updated a README file specifically for the item features and coded the main notebook with two new files, `HM_SNQN_ITEM_FEATURE.ipynb` and `RC_SNQN_ITEM_FEATURE.ipynb`, to run the item features. Additionally, he created a new notebook, `process_RC.ipynb`, to handle category item features and updated `gen_replay_buffer.py` in H&M datasets to get `item_ids.csv`.

Andrew Bonafede: Provided initial implementation of CQL-Loss for SASRec-SA2C model by modifying source code in TRFL package: driver found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/notebooks/SA2C_Alt_Recommender.ipynb), model found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/src/archive/SA2C_vAndrew.py), modified source code found [here](https://github.com/abonafede/trfl/blob/master/trfl/action_value_ops.py). Aided in implementing SNQN CQL-Loss found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/src/SNQN_v2.py). Trained this model on RetailRocket Dataset and added results found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/SNQN_Recommender_Test.ipynb).

Yilun Wu:

Bryce Whitney: Worked on the implementation of CQL-Loss for the SASRec-SNQN model found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/src/SNQN_v2.py). I made additional alterations to the source code to make it compatiable with the H&M dataset, which can be found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/src/SNQN_v2_HM.py). I trained and evaluated the SASRec-SNQN model on the H&M dataset and added the results found [here](https://github.com/szheng3/recommendation-system/blob/main/Explore_CQL/DLR2/HM_SNQN_Recommender.ipynb).

## References

1. Kumar et al., Conservative Q-Learning for Offline Reinforcement Learning, arXiv Aug 2020. (https://arxiv.org/pdf/2006.04779.pdf)
2. Xin Xin et al.,Supervised Advantage Actor-Critic for Recommender Systems, arXiv Nov 2021. (https://arxiv.org/abs/2111.03474)
3. Yifei Ma et al.,Temporal-Contextual Recommendation in Real-Time, Amazon Aug 2020. (https://assets.amazon.science/96/71/d1f25754497681133c7aa2b7eb05/temporal-contextual-recommendation-in-real-time.pdf)
4. Retailrocket, “Retailrocket Recommender System Dataset,” Kaggle, 24-Mar-2017. [Online]. Available: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset. [Accessed: 5-Dec-2022].
5. H&M Group, “H&M personalized fashion recommendations,” Kaggle, 09-May-2022. [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations. [Accessed: 5-Dec-2022].
6. Used gen_replay_buffer.py to generate H&M data from https://github.com/architkaila/recommenders_aipi590