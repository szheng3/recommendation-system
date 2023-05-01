import numpy as np
import tensorflow as tf

# CQL algorithm
class CQL:
    def __init__(self, q_function, policy_network, discount_factor=0.6, alpha=0.5):
        self.q_function = q_function
        self.policy_network = policy_network
        self.discount_factor = discount_factor
        self.alpha = alpha

    def update(self, states, actions, rewards, next_states, is_terminal):
        # Compute the Q-function targets
        q_targets = rewards + np.max(self.q_function.predict(next_states), axis=1) * (1 - is_terminal) * self.discount_factor
    
        # Compute the Q-values for the current states and actions
        q_values = self.q_function.predict(states)
        q_values = np.sum(q_values * actions, axis=1)
    
        # Compute the Q-value constraint
        policy_probs = self.policy_network.predict(states)
        q_constraint = q_values - self.alpha * (np.log(np.sum(np.exp(self.q_function.predict(states) / self.alpha), axis=1)) - np.log(policy_probs + 1e-8))
    
        # Compute the Q-function loss
        q_loss = tf.keras.losses.MSE(q_values, q_targets)
    
        # Compute the policy loss
        entropy = -tf.reduce_mean(policy_probs * tf.math.log(policy_probs + 1e-8))
        policy_loss = -tf.reduce_mean(q_constraint) + 0.1 * entropy
    
        # Compute the total loss
        loss = q_loss + policy_loss
    
        # Update the Q-function
        self.q_function.train_on_batch(states, q_targets)
    
        # Update the policy network
        self.policy_network.train_on_batch(states, actions)
    
        return loss
