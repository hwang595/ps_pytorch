# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate nn on mnist validation data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import time

import sys
import numpy as np
import tensorflow as tf

import mnist

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

start_time = time.time()

def do_eval(saver,
            writer,
            val_acc,
            val_loss,
            images_placeholder,
            labels_placeholder,
            data_set,
            prev_global_step=-1):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  try:
    with tf.Session() as sess:

      # Load checkpoint
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                           ckpt.model_checkpoint_path))
      else:
        print('No checkpoint file found')
        sys.stdout.flush()
        return -1

      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

      # Don't evaluate on the same checkpoint
      if prev_global_step == global_step:
        return prev_global_step

      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
      sys.stdout.flush()

      # Compute accuracy
      num_examples = data_set.num_examples
      feed_dict = mnist.fill_feed_dict(data_set,
                                       images_placeholder,
                                       labels_placeholder,
                                       num_examples)
      acc, loss = sess.run([val_acc, val_loss], feed_dict=feed_dict)

      print('Num examples: %d  Precision @ 1: %f Loss: %f Time: %f' %
            (num_examples, acc, loss, time.time() - start_time))
      sys.stdout.flush()

      # Summarize accuracy
      summary = tf.Summary()
      summary.value.add(tag="Validation Accuracy", simple_value=float(acc))
      summary.value.add(tag="Validation Loss", simple_value=float(loss))
      writer.add_summary(summary, global_step)
    return global_step

  except Exception as e:
    print(e.__doc__)
    print(e.message)

def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Graph creation
    batch_size = dataset.num_examples
    images_placeholder, labels_placeholder = mnist.placeholder_inputs(batch_size)
    logits = mnist.inference(images_placeholder, train=False)
    validation_accuracy = tf.reduce_sum(mnist.evaluation(logits, labels_placeholder)) / tf.constant(batch_size)
    validation_loss = mnist.loss(logits, labels_placeholder)

    # Reference to sess and saver
    sess = tf.Session()
    saver = tf.train.Saver()

    # Create summary writer
    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                           graph_def=graph_def)
    step = -1
    while True:
      step = do_eval(saver, summary_writer, validation_accuracy, validation_loss, images_placeholder, labels_placeholder, dataset, prev_global_step=step)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
