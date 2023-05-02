import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from collections import deque
from utility import pad_history, calculate_hit
from NextItNetModules import *
from SASRecModules import *

import trfl
from trfl import indexing_ops


def parse_args():
    parser = argparse.ArgumentParser(description="Run double q learning.")

    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--r_negative', type=float, default=-0.0,
                        help='reward for the negative behavior.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')
    parser.add_argument('--neg', type=int, default=10,
                        help='number of negative samples.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='number of negative samples.')
    parser.add_argument('--model', type=str, default='GRU',
                        help='the base recommendation models, including GRU,Caser,NItNet and SASRec')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='Number of filters per filter size (default: 16) (for Caser)')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size (for Caser)')
    parser.add_argument('--num_heads', default=1, type=int, help='number heads (for SASRec)')
    parser.add_argument('--num_blocks', default=1, type=int, help='number heads (for SASRec)')
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    return parser.parse_args()


def evaluate(sess):
    eval_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            if evaluated == len(eval_ids):
                break
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                action = id_map[action]

                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                # else:
                #     total_clicks+=1.0
                actions.append(action)
                rewards.append(reward)
                history.append(action)
            evaluated += 1
        prediction = sess.run(QN_1.output2,
                              feed_dict={QN_1.inputs: states, QN_1.len_state: len_states, QN_1.is_training: False})
        sorted_list = np.argsort(prediction)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks,
                      hit_purchase, ndcg_purchase)
    print('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    print('total purchase:%d' % (total_purchase))
    for i in range(len(topk)):
        # hr_click=hit_clicks[i]/total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        # ng_click=ndcg_clicks[i]/total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        # print('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#############################################################')


class QNetwork:
    def __init__(self, hidden_size, learning_rate, item_num, state_size, pretrain, name='DQNetwork'):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.pretrain = pretrain
        self.neg = args.neg
        self.weight = args.weight
        self.model = args.model
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        # self.save_file = save_file
        self.name = name
        with tf.compat.v1.variable_scope(self.name):
            self.all_embeddings = self.initialize_embeddings()
            self.inputs = tf.compat.v1.placeholder(tf.int32,
                                                   [None, state_size])  # sequence of history, [batchsize,state_size]
            self.len_state = tf.compat.v1.placeholder(tf.int32, [
                None])  # the length of valid positions, because short sesssions need to be padded

            # one_hot_input = tf.one_hot(self.inputs, self.item_num+1)
            self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)

            if self.model == 'GRU':
                gru_out, self.states_hidden = tf.compat.v1.nn.dynamic_rnn(
                    tf.compat.v1.nn.rnn_cell.GRUCell(self.hidden_size),
                    self.input_emb,
                    dtype=tf.float32,
                    sequence_length=self.len_state,
                )

            # self.output1 = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num,
            #                                                 activation_fn=None)  # all q-values

            # self.output2= tf.contrib.layers.fully_connected(self.states_hidden, self.item_num,
            #                                                  activation_fn=None, scope="ce-logits")  # all ce logits
            self.output1 = tf.compat.v1.layers.dense(self.states_hidden, self.item_num, activation=None)
            self.output2 = tf.compat.v1.layers.dense(self.states_hidden, self.item_num, activation=None)

            # TRFL way
            self.actions = tf.compat.v1.placeholder(tf.int32, [None])

            self.negative_actions = tf.compat.v1.placeholder(tf.int32, [None, self.neg])

            self.targetQs_ = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.compat.v1.placeholder(tf.float32, [None,
                                                                           item_num])  # used for select best action for double q learning
            self.reward = tf.compat.v1.placeholder(tf.float32, [None])
            self.discount = tf.compat.v1.placeholder(tf.float32, [None])

            self.targetQ_current_ = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            self.targetQ_current_selector = tf.compat.v1.placeholder(tf.float32, [None,
                                                                                  item_num])  # used for select best action for double q learning

            # TRFL double qlearning
            qloss_positive, _ = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount,
                                                      self.targetQs_, self.targetQs_selector)
            neg_reward = tf.constant(reward_negative, dtype=tf.float32, shape=(args.batch_size,))
            qloss_negative = 0
            for i in range(self.neg):
                negative = tf.gather(self.negative_actions, i, axis=1)

                qloss_negative += trfl.double_qlearning(self.output1, negative, neg_reward,
                                                        self.discount, self.targetQ_current_,
                                                        self.targetQ_current_selector)[0]

            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)

            self.loss = tf.reduce_mean(self.weight * (qloss_positive + qloss_negative) + ce_loss)
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def initialize_embeddings(self):
        all_embeddings = dict()
        if self.pretrain == False:
            with tf.compat.v1.variable_scope(self.name):
                state_embeddings = tf.Variable(tf.random.normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                               name='state_embeddings')
                pos_embeddings = tf.Variable(tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                             name='pos_embeddings')
                all_embeddings['state_embeddings'] = state_embeddings
                all_embeddings['pos_embeddings'] = pos_embeddings

        return all_embeddings


if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    reward_negative = args.r_negative
    topk = [5, 10, 15, 20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    tf.compat.v1.reset_default_graph()

    QN_1 = QNetwork(name='QN_1', hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                    state_size=state_size, pretrain=False)
    QN_2 = QNetwork(name='QN_2', hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                    state_size=state_size, pretrain=False)

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    # saver = tf.train.Saver()

    ids_df = pd.read_csv(os.path.join(data_directory, 'item_ids.csv'))
    ids_df['new_id'] = range(len(ids_df))
    id_map = ids_df.set_index('itemid')['new_id'].to_dict()

    total_step = 0
    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        # evaluate(sess)
        num_rows = replay_buffer.shape[0]
        num_batches = int(num_rows / args.batch_size)
        for i in range(args.epoch):
            for j in range(num_batches):
                batch = replay_buffer.sample(n=args.batch_size).to_dict()

                # state = list(batch['state'].values())

                next_state = list(batch['next_state'].values())
                next_state = [id_map[x] for x in next_state]
                len_next_state = list(batch['len_next_states'].values())
                # double q learning, pointer is for selecting which network  is target and which is main
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = QN_1
                    target_QN = QN_2
                else:
                    mainQN = QN_2
                    target_QN = QN_1
                target_Qs = sess.run(target_QN.output1,
                                     feed_dict={target_QN.inputs: next_state,
                                                target_QN.len_state: len_next_state,
                                                target_QN.is_training: True})
                target_Qs_selector = sess.run(mainQN.output1,
                                              feed_dict={mainQN.inputs: next_state,
                                                         mainQN.len_state: len_next_state,
                                                         mainQN.is_training: True})
                # Set target_Qs to 0 for states where episode ends
                is_done = list(batch['is_done'].values())
                for index in range(target_Qs.shape[0]):
                    if is_done[index]:
                        target_Qs[index] = np.zeros([item_num])

                state = list(batch['state'].values())
                state = [id_map[x] for x in state]

                len_state = list(batch['len_state'].values())
                target_Q_current = sess.run(target_QN.output1,
                                            feed_dict={target_QN.inputs: state,
                                                       target_QN.len_state: len_state,
                                                       target_QN.is_training: True})
                target_Q__current_selector = sess.run(mainQN.output1,
                                                      feed_dict={mainQN.inputs: state,
                                                                 mainQN.len_state: len_state,
                                                                 mainQN.is_training: True})
                action = list(batch['action'].values())
                action = [id_map[x] for x in action]

                negative = []

                for index in range(target_Qs.shape[0]):
                    negative_list = []
                    for i in range(args.neg):
                        neg = np.random.randint(item_num)
                        while neg == action[index]:
                            neg = np.random.randint(item_num)
                        negative_list.append(neg)
                    negative.append(negative_list)

                is_buy = list(batch['is_buy'].values())
                reward = []
                for k in range(len(is_buy)):
                    reward.append(reward_buy if is_buy[k] == 1 else reward_click)
                discount = [args.discount] * len(action)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs: state,
                                              mainQN.len_state: len_state,
                                              mainQN.targetQs_: target_Qs,
                                              mainQN.reward: reward,
                                              mainQN.discount: discount,
                                              mainQN.actions: action,
                                              mainQN.targetQs_selector: target_Qs_selector,
                                              mainQN.negative_actions: negative,
                                              mainQN.targetQ_current_: target_Q_current,
                                              mainQN.targetQ_current_selector: target_Q__current_selector,
                                              mainQN.is_training: True
                                              })
                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 4000 == 0:
                    evaluate(sess)
