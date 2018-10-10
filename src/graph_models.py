import tensorflow as tf
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from lib.models import *


class multi_cgcnn(cgcnn):
    """
    Graph CNN with multiple features per node. Input features are taken through multiple graphs and the last layers
    are concatenated before the final output. This class is inherited from cgcnn; it over-writes the inference method
    and some methods from base_model. The number of features/graphs is specified by n_graph.
    """

    def __init__(self, L, F, K, p, M, n_graph, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name=''):
        self.n_graph = n_graph
        super().__init__(L, F, K, p, M, filter, brelu, pool,
                         num_epochs, learning_rate, decay_rate, decay_steps, momentum,
                         regularization, dropout, batch_size, eval_frequency,
                         dir_name)

    def _inference_single(self, x, dropout, j):
        """
        Inference on a single graph.

        :param x: (2d tensor) data on a single graph of the shape [example, node]
        :param dropout: (float) keep probability between 0 and 1
        :param j: (int) graph index for variable scope keeping
        :return: (tensorflow) fully-connected layer of a single graph
        """
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)
        for i in range(len(self.p)):
            with tf.variable_scope('graph{}_'.format(j) + 'conv{}'.format(i + 1)):  # variable scope keeping
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])

        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M * F)])  # N x M
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('graph{}_'.format(j) + 'fc{}'.format(i + 1)):  # variable scope keeping
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        # with tf.variable_scope('logits'):
        #     x = self.fc(x, self.M[-1], relu=False)
        return x

    def _inference(self, x, dropout):
        """
        Inference on multiple graphs.

        :param x: (3d tensor) data on multiple graphs of the shape [example, node, feature]
        :param dropout: (float) keep probability between 0 and 1
        :return: (tensorflow) final output of multiple graphs
        """
        hidden = []
        for i in range(self.n_graph):
            hidden.append(self._inference_single(x[:, :, i], dropout, i + 1))  # slice x to the right shape
        x = tf.concat(hidden, 1)
        x = self.fc(x, self.M[-1], relu=False)
        return x

    # High-level interface which runs the constructed computational graph.

    def predict(self, data, labels=None, sess=None):
        """
        Over-write to use 3d data of the shape [example, node, feature].
        """
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))  # 3d data
            tmp_data = data[begin:end, :, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def fit(self, train_data, train_labels, val_data, val_labels):
        """
        Over-write to use 3d train_data of the shape [example, node, feature].
        """
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps + 1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx, :, :], train_labels[idx]  # train_data is 3d
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    # Methods to construct the computational graph.

    def build_graph(self, M_0):
        """
        Build the computational graph of the model.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, self.n_graph), 'data')  # 3d ph_data
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()
