import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of
    B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances
    """
    # with tf.variable_scope('pairwise_dist'):
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb + 1e-20,
                           0.0))
    return D

def label_leakage_roc_auc(y, predicted_value):
    if np.sum(y) == 0: # no positive examples in this batch
        return np.float32(0.5)
    val_max = np.amax(predicted_value)
    val_min = np.amin(predicted_value)
    pred = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)
    # print("y: {}".format(y))
    # print("pred: {}".format(pred))
    auc = roc_auc_score(y_true=y, y_score=pred)
    return np.float32(auc)


def hint_attack(labels, grads, num_hints = 1):
    labels = tf.reshape(labels, shape=[tf.shape(grads)[0]])
    def no_positive_instance():
        print("no positive instance in this batch, leak_auc_roc = 0.5 by default")
        return 0.5
    def compute_auc_score(labels, grads, num_hints):
        pos_grads = tf.boolean_mask(grads, labels)
        pos_grads = tf.random.shuffle(pos_grads)
        # print("pos_grads: {}".format(pos_grads))
        # print("num_hints: {}".format(tf.minimum(num_hints, tf.shape(pos_grads)[0])))
        selected_grads = tf.slice(pos_grads, [0, 0], [tf.minimum(num_hints, tf.shape(pos_grads)[0]), -1])
        # print("selected_grads: {}".format(selected_grads))
        dist_res = pairwise_dist(grads, selected_grads)
        # print("dist_res: {}".format(dist_res))
        # print("dist_res shape: {}".format(tf.shape(dist_res)))
        dist_res = tf.math.reduce_min(dist_res, axis = 1)
        # print("dist_res reduce : {}".format(dist_res))
        dist_res = tf.reshape(dist_res, shape=[tf.shape(grads)[0]])
        auc_score = label_leakage_roc_auc(labels, -1.0 * dist_res)
        return auc_score
    auc_score = tf.cond(tf.cast(tf.reduce_sum(labels), dtype=tf.float32)>= 1.0, lambda: compute_auc_score(labels, grads, num_hints), lambda: no_positive_instance())
    return auc_score


# a = tf.Variable([[1,1,1,1],[2,2,2,2], [3,3,3,3], [4,4,4,4],[5,5,5,5]], dtype=tf.float32)
# print(a)
# # b = tf.Variable([[1,1,1,1], [2,2,2,2]], dtype=tf.float32)
# # print(b)
# labels = tf.Variable([1,1,0,1,0])
# print(labels)
# score = hint_attack(labels, a, num_hints = 1)
# print(score)