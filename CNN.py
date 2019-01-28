import sys
sys.path.append('/u/tyang21/Documents/libs')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import pandas as pd


def parse_data(train_path, test_path):
    '''parse data from certain paths'''
    train_df = pd.DataFrame.from_csv(train_path)
    test_df = pd.DataFrame.from_csv(test_path, index_col=None)

    train_num = len(train_df.index)
    test_num = len(test_df.index)
    row_len = 28
    col_len = 28

    train_labels = train_df.index.values
    test_labels = test_df.index.values
    train_labels.astype(np.float32)
    test_labels.astype(np.float32)

    train_images = train_df.values.reshape([train_num, 28, 28]).astype(np.float32)
    test_images = test_df.values.reshape([test_num, 28, 28]).astype(np.float32)
    train = (train_images[:40000], train_labels[:40000])
    test = (test_images, test_labels)
    return train, test


def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    """convolution layer"""
    with tf.variable_scope(scope_name) as scope:
        in_dim = inputs.shape[-1]
        kernel = tf.get_variable('kernel', [k_size,k_size,in_dim,filters], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        conv = tf.nn.conv2d(inputs, filter=kernel, strides=[1, stride, stride, 1], padding=padding )
    return tf.nn.relu(conv+bias, name=scope_name)


def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    """Pooling layer"""
    with tf.variable_scope(scope_name) as scope:
        pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)
    return pool


def fully_connected(inputs, out_dim, scope_name='fc'):
    """Full connected layer"""
    with tf.variable_scope(scope_name) as scope:
        in_dim = inputs.shape[-1]
        W = tf.get_variable('W', shape=[in_dim, out_dim], initializer=tf.truncated_normal_initializer())
        B = tf.get_variable('B', shape=[out_dim], initializer= tf.truncated_normal_initializer())
    return tf.matmul(inputs, W)+B


class ConvNet():
    def __init__(self):
        self.train_path = './all/train.csv'
        self.test_path = './all/test.csv'
        self.keep_prob = tf.constant(0.75)
        self.lr = 0.001
        self.skip_step = 10
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')

    def get_data(self):
        """Get data transformed in tensorflow form"""
        with tf.name_scope('data'):
            train, test = parse_data(self.train_path, self.test_path)

            train_data = tf.data.Dataset.from_tensor_slices(train)
            test_data = tf.data.Dataset.from_tensor_slices(test)

            train_data = train_data.shuffle(10000)
            train_data = train_data.batch(4000)
            test_data = test_data.batch(4000)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, label = iterator.get_next()
            self.label = tf.one_hot(indices=label, depth=10, on_value=1, off_value=0)
            self.img = tf.reshape(img, [-1, 28, 28, 1])

            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)
            print('Get data done')

    def inference(self):
        """Structure of the graph"""
        conv1 = conv_relu(self.img, filters=32, k_size= 5, stride=1, padding='SAME',scope_name= 'conv1')
        pool1 = maxpool(conv1, ksize=2, stride=1, padding='SAME', scope_name='pool1')
        conv2 = conv_relu(pool1, filters=64, k_size=5, stride=1, padding='SAME', scope_name='conv2')
        pool2 = maxpool(conv2, ksize=2, stride=1, padding='SAME', scope_name='pool2')
        feature_shape = pool2.shape[1]*pool2.shape[2]*pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_shape])
        fc1 = fully_connected(pool2, 1024, scope_name='fc1')
        dropout = tf.nn.dropout(tf.nn.relu(fc1), self.keep_prob, name='relu_dropout')
        self.logits = fully_connected(dropout, 10, scope_name='logits')
        print('Inference done')

    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')
        print('Loss done')

    def optimize(self):
        self.optimizor = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        print('Opt done')

    def eval(self):
        with tf.name_scope('predict'):
            pred = tf.nn.softmax(self.logits)
            prediction = tf.argmax(pred, 1)
            self.pred = prediction
            correct_pres = tf.equal(tf.argmax(self.label, 1), prediction)
            self.acc = tf.reduce_sum(tf.cast(correct_pres, tf.float32))
        print('Eval done')

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()

    def train_one_step(self, sess, init, epoch, step):
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([self.optimizor, self.loss])
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        # saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet')
        print('Average loss at epoch {0}:{1}'.format(epoch, total_loss/n_batches))
        return step

    def eval_one_step(self, sess, init, epoch, step):
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run([self.acc])
                total_correct_preds += accuracy_batch[0]
        except tf.errors.OutOfRangeError:
            pass
        print('Total accuracy at epoch {0}:{1}'.format(epoch, total_correct_preds))

    def output(self, sess, init, filename = './result'):
        sess.run(init)
        tag = 0
        try:
            while True:
                filename = './result' + str(tag) + '.csv'
                prediction = sess.run([self.pred])
                pd.DataFrame(prediction).to_csv(filename)
                tag += 1
        except tf.errors.OutOfRangeError:
            pass

    def train(self, n_epochs):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
            os.mkdir('./checkpoints/convnet_mnist')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                step = self.train_one_step(sess,self.train_init, epoch, step)
                if epoch % 10 == 0:
                    self.eval_one_step(sess, self.test_init, epoch, step)
            saver.save(sess, './ckpt/model.ckpt')
            saver.restore(sess,'./ckpt/model.ckpt' )
            self.output(sess, self.test_init)


if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs = 30)
