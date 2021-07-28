import tensorflow as tf
import cv2


def conv_layer(name, X, in_channels, out_filters, ksize, stride, trainable=True, padding="SAME",
               with_elu=True,
               with_bn=True, decay=0.9, epsilon=1e-5):
    print(X.shape)
    if not hasattr(ksize, "__len__"):
        ksize = [ksize, ksize]
    if not hasattr(stride, "__len__"):
        stride = [stride, stride]
    with tf.variable_scope(name):
        w = tf.get_variable("weight", [ksize[0], ksize[1], in_channels, out_filters], tf.float32,
                            # tf.truncated_normal_initializer(0.0, 0.01),
                            tf.contrib.layers.xavier_initializer(),
                            trainable=trainable)
        b = tf.get_variable("bias", [out_filters], tf.float32,
                            tf.contrib.layers.xavier_initializer(),  # tf.constant_initializer(0.01),
                            trainable=trainable)
        Y = tf.add(tf.nn.conv2d(
            X, w, [1, stride[0], stride[1], 1], padding), b)
        if with_bn:
            Y = tf.contrib.layers.batch_norm(Y, decay=decay, epsilon=epsilon,
                                             updates_collections=None,
                                             center=True, scale=True,
                                             trainable=True,
                                             is_training=trainable)
        if with_elu:
            Y = tf.nn.elu(Y)
    return Y, w, b


def max_pool(name, X, ksize, padding="SAME"):
    with tf.variable_scope(name):
        Y = tf.nn.max_pool(X, [1, ksize, ksize, 1], [
                           1, ksize, ksize, 1], padding)
    return Y


def fc_layer(name, X, in_channels, out_filters, trainable=True,
             with_elu=True,
             with_bn=True, decay=0.9, epsilon=1e-5):
    with tf.variable_scope(name):
        w = tf.get_variable("weight", [in_channels, out_filters], tf.float32,
                            # tf.truncated_normal_initializer(0.0, 0.01),
                            tf.contrib.layers.xavier_initializer(),
                            trainable=trainable)
        b = tf.get_variable("bias", [out_filters], tf.float32,
                            tf.contrib.layers.xavier_initializer(),  # tf.constant_initializer(0.01),
                            trainable=trainable)
        Y = tf.add(tf.matmul(X, w), b)
        if with_bn:
            Y = tf.contrib.layers.batch_norm(Y, decay=decay, epsilon=epsilon,
                                             updates_collections=None,
                                             center=True, scale=True,
                                             trainable=True,
                                             is_training=trainable)
        if with_elu:
            Y = tf.nn.elu(Y)
    return Y, w, b

def preprocess(in_img):
    im = cv2.cvtColor(in_img, cv2.COLOR_BGR2YUV)

    return (im/127.5 - 1.0)

# def preprocess(cv_mat_im, crop):
#     #im = cv2.cvtColor(cv2.resize(cv_mat_im, (500, 300), cv2.INTER_AREA), cv2.COLOR_BGR2YUV)
#     im = cv2.cvtColor(cv_mat_im, cv2.COLOR_BGR2YUV)
#     return (im/127.5 - 1.0)


def build_net(trainable=True):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    X = tf.placeholder(tf.float32, [600, 1000, 3], "input")                   # 3@66x200    
    x_next = tf.reshape(X,[1, 600, 1000, 3])
    # no padding, 5x5 kernel, 2x2 stride
    conv1, w1, b1 = conv_layer("conv1",     x_next,    3,   64, 5, 2, padding="SAME", trainable=trainable, with_bn=True) # 24@31x98
    conv2, w2, b2 = conv_layer("conv2", conv1,   64,   64, 5, 2, padding="SAME", trainable=trainable, with_bn=True) # 36@14x47
    conv3, w3, b3 = conv_layer("conv3", conv2,   64,   64, 5, 2, padding="SAME", trainable=trainable, with_bn=True) # 48@5x22
    # no padding, 3x3 kenerl, 1x1 stride
    conv4, w4, b4 = conv_layer("conv4", conv3,   64,   64, 5, 2, padding="SAME", trainable=trainable, with_bn=True) # 64@3x20
    conv5, w5, b5 = conv_layer("conv5", conv4,   64,   64, 5, 2, padding="SAME", trainable=trainable, with_bn=True) # 64@1x18
    
    #dp = tf.nn.dropout(conv5, keep_prob)
    dp = conv5

    # fc layers
    fc1,   w6, b6 = conv_layer("fc1", dp,    64, 1000, [19, 32], 1, padding="VALID", trainable=trainable, with_bn=True)    # 1164@1x1
    fc2,   w7, b7 = conv_layer("fc2", fc1, 1000,  100, 1, 1, padding="VALID", trainable=trainable, with_bn=True)          # 100@1x1
    fc3,   w8, b8 = conv_layer("fc3", fc2,  100,   50, 1, 1, padding="VALID", trainable=trainable, with_bn=True)          # 50@1x1
    fc4,   w9, b9 = conv_layer("fc4", fc3,   50,   10, 1, 1, padding="VALID", trainable=trainable, with_bn=True)          # 10@1x1
    fc5, w10, b10 = conv_layer("fc5", fc4,   10,    1, 1, 1, padding="VALID", with_elu=False, with_bn=False, trainable=trainable)                           # 1@1x1

    with tf.variable_scope("predict"):
        y = tf.reshape(fc5, [-1])

    return X, keep_prob, y#, (w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w9, b9, w10, b10)


def build_loss(Y, y, regularizer_factor, regularized_vars):
    with tf.variable_scope("mse_loss"):
        mse = tf.reduce_mean(tf.square(Y - y))
    with tf.variable_scope("l2_regularizer"):
        regularizer = tf.add_n([tf.nn.l2_loss(var)for var in regularized_vars])
    with tf.variable_scope("cost"):
        loss = mse + regularizer_factor*regularizer

    return loss
