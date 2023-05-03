import trfl
import tensorflow as tf


def compute_cql_loss(target_q_values, policy_q_values):
    """
    Compute CQL loss.
    
    Args:
        target_q_values (tf.Tensor): All Q-values for all actions.
        policy_q_values (tf.Tensor): Q-values for the selected actions from the main network.
        
    
    Returns:
        cql_loss (tf.Tensor): The CQL loss tensor.
    """

    # Computes the logsumexp of target Q-values
    logsumexp_target_q_values = tf.reduce_logsumexp(target_q_values, axis=1)

    # Computes the difference between logsumexp of target Q-values and policy Q-values
    q_value_diff = logsumexp_target_q_values - policy_q_values

    # Computes CQL loss
    cql_loss = tf.reduce_mean(q_value_diff)

    return cql_loss
