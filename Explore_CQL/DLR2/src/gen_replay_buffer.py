import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from utility_v2 import to_pickled_df, pad_history


def parse_args():
    '''
        Function to parse the given command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Generate replay buffer data.")

    parser.add_argument('--data', nargs='?', default='data',
                        help='Data Directory.')

    parser.add_argument('--state_len', type=int, default=10,
                        help='Max state length.')

    parser.add_argument('--seed', type=int, default=1234,
                        help='Value of random seed.')

    parser.add_argument('--format', choices=['paper', 'csv'], default='paper',
                        help='Select output format: "paper" or "csv".')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DATA = args.data

    print("\nStarting to pre-process data...")
    # Read in events log
    event_df = pd.read_csv(os.path.join(DATA, 'events.csv'), header=0)
    event_df.columns = ['timestamp','session_id','behavior','item_id','transid']
    # Remove transid column
    event_df =event_df[event_df['transid'].isnull()]
    event_df = event_df.drop('transid',axis=1)
    # Remove users with <=2 interactions
    event_df['valid_session'] = event_df.session_id.map(event_df.groupby('session_id')['item_id'].size() > 2)
    event_df = event_df.loc[event_df.valid_session].drop('valid_session', axis=1)
    # Remove items with <=2 interactions
    event_df['valid_item'] = event_df.item_id.map(event_df.groupby('item_id')['session_id'].size() > 2)
    event_df = event_df.loc[event_df.valid_item].drop('valid_item', axis=1)
    # Transform to ids using LabelEncoders
    item_encoder = LabelEncoder()
    session_encoder= LabelEncoder()
    behavior_encoder=LabelEncoder()
    event_df['item_id'] = item_encoder.fit_transform(event_df.item_id)
    event_df['session_id'] = session_encoder.fit_transform(event_df.session_id)
    event_df['behavior']=behavior_encoder.fit_transform(event_df.behavior)
    # Sort data by user and timestamp
    print("\nSorting and pickling data...")
    event_df['is_buy'] = 1 - event_df['behavior']
    event_df = event_df.drop('behavior', axis=1)
    sorted_events = event_df.sort_values(by=['session_id', 'timestamp'])
    # Store and pickle sorted data
    sorted_events.to_csv(os.path.join(DATA, 'sorted_events.csv'), index=None, header=True)
    to_pickled_df(DATA, sorted_events=sorted_events)

    # Get unique sessions ids and shuffle order
    total_sessions = sorted_events.session_id.unique()
    # Set random seed before shuffling
    np.random.seed(args.seed)
    np.random.shuffle(total_sessions)

    # Split data into train/test sets
    # Split ratio: 80% train, 10% validation, and 10% test
    print("\nSplitting data into train, validation, and test sets...")
    split_ratio = np.array([0.8, 0.1, 0.1])
    train_ids, val_ids, test_ids = np.array_split(total_sessions, (split_ratio[:-1].cumsum() * len(total_sessions)).astype(int))
    train_sessions = sorted_events[sorted_events['session_id'].isin(train_ids)]
    val_sessions = sorted_events[sorted_events['session_id'].isin(val_ids)]
    test_sessions = sorted_events[sorted_events['session_id'].isin(test_ids)]
    # Pickle split data as dataframes
    print("\nPickling train, validation, and test sets...")
    to_pickled_df(DATA, sampled_train=train_sessions)
    to_pickled_df(DATA, sampled_val=val_sessions)
    to_pickled_df(DATA,sampled_test=test_sessions)

    # Count item popularity
    # Store popularity in a dictionary
    print("\nCalculating item popularity and storing as dictionary...")
    total_actions = sorted_events.shape[0]
    pop_dict = {}
    for index, row in sorted_events.iterrows():
        action=row['item_id']
        if action in pop_dict:
            pop_dict[action]+=1
        else:
            pop_dict[action]=1
    for key in pop_dict:
        pop_dict[key]=float(pop_dict[key])/float(total_actions)
    # Save popularity info to a local file
    with open(os.path.join(DATA, 'pop_dict.txt'), 'w') as f:
        f.write(str(pop_dict))

    # Generate replay buffer
    # Pad item or trim item if length is too short or too long
    print("\nGenerating replay buffer from train set...")
    STATE_LEN = args.state_len
    item_ids = sorted_events.item_id.unique()
    pad_item = len(item_ids)
    train_sessions = pd.read_pickle(os.path.join(DATA, 'sampled_train.df'))
    groups = train_sessions.groupby('session_id')
    ids = train_sessions.session_id.unique()
    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [], [], []
    for id in ids:
        group = groups.get_group(id)
        history = []
        for index, row in group.iterrows():
            s = list(history)
            len_state.append(STATE_LEN if len(s) >= STATE_LEN else 1 if len(s) == 0 else len(s))
            s = pad_history(s, STATE_LEN, pad_item)
            a = row['item_id']
            is_b = row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s = list(history)
            len_next_state.append(STATE_LEN if len(next_s) >= STATE_LEN else 1 if len(next_s) == 0 else len(next_s))
            next_s = pad_history(next_s, STATE_LEN, pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1]=True
    # Store replay buffer in dictionary and save as pickled dataframe
    dic1 = {
        'state': state,
        'len_state': len_state,
        'action': action,
        'is_buy': is_buy,
        'next_state': next_state,
        'len_next_states': len_next_state,
        'is_done': is_done
    }
    print("\nPickling replay buffer...")
    replay_buffer = pd.DataFrame(data=dic1)
    to_pickled_df(DATA, replay_buffer=replay_buffer)
    # Store replay buffer as CSV format if specifically requested
    if args.format == 'csv':
        print("\Saving replay buffer as CSV...")
        replay_buffer.to_csv(os.path.join(DATA, 'replay_buffer.csv'), index=None, header=True)
    # Store data statistics in dictionary and save as pickled dataframe
    dic2 = {
        'state_size': [STATE_LEN],
        'item_num': [pad_item]
    }
    print("\nPickling data statistics...")
    data_statis = pd.DataFrame(data=dic2)
    to_pickled_df(DATA, data_statis=data_statis)

    # End script
    print("\nScript completed successfully!")