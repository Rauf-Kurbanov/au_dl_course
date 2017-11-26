import tensorflow as tf
import tensorflow.contrib.layers as layers
from model_components import task_specific_attention, bidirectional_rnn
from lazy import lazy


class HanSequenceLabellingModel():
    def __init__(self,
                 embedding_size,
                 classes,
                 word_cell,
                 sentence_cell,
                 word_output_size,
                 sentence_output_size,
                 max_grad_norm,
                 dropout_keep_proba,
                 is_training=None,
                 learning_rate=1e-4,
                 scope=None):
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_proba = dropout_keep_proba
        self.is_training = is_training
        self.scope = scope
        self.learning_rate = learning_rate

        self.create_placeholders()
        self.word_level_output
        self.sentence_level_output
        self.prediction
        self.train_op
        self.summary

    def create_placeholders(self):
        with tf.name_scope("placeholders"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if self.is_training is not None:
                self.is_training = self.is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

            # [document x sentence x word x embed_size]
            self.inputs_embedded = tf.placeholder(shape=(None, None, None, self.embedding_size),
                                                  dtype=tf.float32, name='inputs')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')

            (self.document_size,
             self.sentence_size,
             self.word_size, _) = tf.unstack(tf.shape(self.inputs_embedded))

    @lazy
    def word_level_output(self):
        with tf.name_scope("word_level"):
            word_level_inputs = tf.reshape(self.inputs_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.document_size * self.sentence_size])

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell, self.word_cell,
                    word_level_inputs, word_level_lengths,
                    scope=scope)

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        self.word_output_size,
                        scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

        return word_level_output

    @lazy
    def sentence_level_output(self):
        with tf.name_scope("sentence_level"):
            sentence_inputs = tf.reshape(
                self.word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

            with tf.variable_scope('sentence') as scope:
                sentence_encoder_output, _ = bidirectional_rnn(
                    self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output = task_specific_attention(
                        sentence_encoder_output, self.sentence_output_size, scope=scope)

                with tf.variable_scope('dropout'):
                    sentence_level_output = layers.dropout(
                        sentence_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )
            return sentence_level_output

    @lazy
    def prediction(self):
        with tf.variable_scope('classifier'):
            self.logits = layers.fully_connected(
                self.sentence_level_output, self.classes, activation_fn=None)

            prediction = tf.argmax(self.logits, axis=-1)

        return prediction

    @lazy
    def train_op(self):
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()
            grads, self.global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

        return train_op

    @lazy
    def summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('global_grad_norm', self.global_norm)
            summary_op = tf.summary.merge_all()

        return summary_op


def test():
    try:
        from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
    except ImportError:
        GRUCell = tf.nn.rnn_cell.GRUCell

    tf.reset_default_graph()
    with tf.Session() as session:
        model = HanSequenceLabellingModel(
            # vocab_size=10,
            embedding_size=3,
            classes=2,
            word_cell=GRUCell(10),
            sentence_cell=GRUCell(10),
            word_output_size=10,
            sentence_output_size=10,
            max_grad_norm=5.0,
            dropout_keep_proba=0.5,
        )
        session.run(tf.global_variables_initializer())

        fd = {
            model.is_training: True,
            model.inputs_embedded: [[
                [[5, 0, 0], [4, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[3, 0, 0], [3, 0, 0], [6, 0, 0], [7, 0, 0]],
                [[6, 0, 0], [7, 0, 0], [0, 0, 0], [0, 0, 0]]
            ],
                [
                    [[2, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]],
                    [[3, 0, 0], [3, 0, 0], [6, 0, 0], [7, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]],
            model.word_lengths: [
                [3, 4, 2],
                [3, 4, 0],
            ],
            model.sentence_lengths: [3, 2],
            model.labels: [0, 1],
            model.sample_weights: [1, 1],
        }

        session.run(model.train_op, fd)


if __name__ == '__main__':
    test()
