# zidingyi loss
import tensorflow as tf


def loss_smooth_l1(pred, targets, diff_weight=1., sigma=1.0):
    sigma_2 = sigma ** 2
    diff = pred - targets
    # cha jiaru yige quanzhong
    diff_weighted = diff * diff_weight
    diff_weighted_abs = tf.abs(diff_weighted)
    # x yu 1. / sigma_2 bijiao (1. / sigma_2 wei yuzhi)
    sign_smoothl1 = tf.stop_gradient(tf.to_float(tf.less(diff_weighted_abs, 1. / sigma_2)))
    loss_a_unit = tf.pow(diff_weighted, 2) * (sigma_2 / 2.) * sign_smoothl1 + \
                  (diff_weighted_abs - (0.5 / sigma_2)) * (1. - sign_smoothl1)
    loss_total = tf.reduce_mean(tf.reduce_sum(loss_a_unit))
    return loss_total

# import numpy as np
# pred = np.asarray([[1.,2],[3,4]],dtype=np.float64)
# targets = np.asarray([[1,2], [5.,3]], dtype=np.float64)
# loss = loss_smooth_l1(pred, targets)
#
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(loss))
