"""
@author: Thang Nguyen <nhthang1009@gmail.com>
"""
import tensorflow as tf


class Char_level_cnn(object):
    def __init__(self, batch_size=128, num_classes=14, feature="small",
                 kernel_size=[7, 7, 3, 3, 3, 3], padding="VALID"):
        super(Char_level_cnn, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        if feature == "small":
            self.num_filters = 256
            self.stddev_initialization = 0.05
            self.num_fully_connected_features = 1024
        else:
            self.num_filters = 1024
            self.stddev_initialization = 0.02
            self.num_fully_connected_features = 2048
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, input, keep_prob):

        output = tf.expand_dims(input, -1)
        output = self._create_conv(output, [output.get_shape().as_list()[1], self.kernel_size[0], 1, self.num_filters],
                                   "conv1",
                                   3)
        output = self._create_conv(output, [1, self.kernel_size[1], self.num_filters, self.num_filters], "conv2", 3)
        output = self._create_conv(output, [1, self.kernel_size[2], self.num_filters, self.num_filters], "conv3")
        output = self._create_conv(output, [1, self.kernel_size[3], self.num_filters, self.num_filters], "conv4")
        output = self._create_conv(output, [1, self.kernel_size[4], self.num_filters, self.num_filters], "conv5")
        output = self._create_conv(output, [1, self.kernel_size[5], self.num_filters, self.num_filters], "conv6", 3)

        new_feature_size = int(self.num_filters * ((input.get_shape().as_list()[2] - 96) / 27))
        flatten = tf.reshape(output, [-1, new_feature_size])

        output = self._create_fc(flatten, [new_feature_size, self.num_fully_connected_features], "fc1", keep_prob)
        output = self._create_fc(output, [self.num_fully_connected_features, self.num_fully_connected_features], "fc2",
                                 keep_prob)
        output = self._create_fc(output, [self.num_fully_connected_features, self.num_classes], "fc3")

        return output

    def _create_conv(self, input, shape, name_scope, pool_size=None):
        with tf.name_scope(name_scope):
            weight = self._initialize_weight(shape, self.stddev_initialization)
            bias = self._initialize_bias([shape[-1]])
            conv = tf.nn.conv2d(input=input, filter=weight, strides=[1, 1, 1, 1], padding=self.padding, name='conv')
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            if pool_size:
                return tf.nn.max_pool(value=activation, ksize=[1, 1, pool_size, 1], strides=[1, 1, pool_size, 1],
                                      padding=self.padding, name='maxpool')
            else:
                return activation

    def _create_fc(self, input, shape, name_scope, keep_prob=None):
        with tf.name_scope(name_scope):
            weight = self._initialize_weight(shape, self.stddev_initialization)
            bias = self._initialize_bias([shape[-1]])
            dense = tf.nn.bias_add(tf.matmul(input, weight), bias, name="dense")
            if keep_prob is not None:
                return tf.nn.dropout(dense, keep_prob, name="dropout")
            else:
                return dense

    def _initialize_weight(self, shape, stddev):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32, name='weight'))

    def _initialize_bias(self, shape):
        return tf.Variable(tf.constant(0, shape=shape, dtype=tf.float32, name='bias'))

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def accuracy(self, logits, labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), dtype=tf.float32))

    def confusion_matrix(self, logits, labels):
        return tf.confusion_matrix(tf.cast(labels, tf.int64), tf.argmax(logits, 1), num_classes=self.num_classes)
