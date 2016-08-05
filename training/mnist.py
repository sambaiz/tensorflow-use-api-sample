import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Mnist:

    def __init__(self):

        g = tf.Graph()

        with g.as_default():

            W_conv1 = self._weight_variable([5, 5, 1, 32],  "W_conv1")
            b_conv1 = self._bias_variable([32],  "b_conv1")

            self._x = tf.placeholder(tf.float32, [None, 784])
            x_image = tf.reshape(self._x, [-1,28,28,1])

            h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            W_conv2 = self._weight_variable([5, 5, 32, 64],  "W_conv2")
            b_conv2 = self._bias_variable([64],  "b_conv2")

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            W_fc1 = self._weight_variable([7 * 7 * 64, 1024],  "W_fc1")
            b_fc1 = self._bias_variable([1024],  "b_fc1")

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            self._keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

            W_fc2 = self._weight_variable([1024, 10],  "W_fc2")
            b_fc2 = self._bias_variable([10],  "b_fc2")

            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            self._what_number = tf.argmax(y_conv, 1)

            self._y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._y_ * tf.log(y_conv), reduction_indices=[1]))
            self._train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self._y_,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.sess = tf.Session()
            init = tf.initialize_all_variables()
            self.sess.run(init)
            self._saver = tf.train.Saver()

    def save(self, ckpt_file_name):
        self._saver.save(self.sess, ckpt_file_name)

    def restore(self, ckpt_file_name):
        self._saver.restore(self.sess, ckpt_file_name)

    def what_number(self, image_array):
        return self.sess.run(self._what_number, feed_dict={self._x: image_array, self._keep_prob: 1.0})

    def train(self, num):
        if not hasattr(self, "_mnist"):
            self._mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        for i in range(num):
            batch = self._mnist.train.next_batch(50)
            if i%100 == 0:
              train_accuracy = self._accuracy.eval(session=self.sess, feed_dict={
                      self._x:batch[0], self._y_: batch[1], self._keep_prob: 1.0
                  })
              print("step %d, training accuracy %g"%(i, train_accuracy))
            self.sess.run(self._train_step, feed_dict={self._x: batch[0], self._y_: batch[1], self._keep_prob: 0.5})

        print("test accuracy %g"%self._accuracy.eval(session=self.sess, feed_dict={
                self._x: self._mnist.test.images, self._y_: self._mnist.test.labels, self._keep_prob: 1.0
            }))

    def close(self):
        self.sess.close()

    def _weight_variable(self, shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

    def _conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
