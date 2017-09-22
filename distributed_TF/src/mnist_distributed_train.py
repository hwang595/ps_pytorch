"""
Train distributed mnist
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import distributed_train
import mnist_data

FLAGS = tf.app.flags.FLAGS

def main(unused_args):
  assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  # Extract all the hostnames for the ps and worker jobs to construct the
  # cluster spec.
  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts)

  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                       'worker': worker_hosts})
  server = tf.train.Server(
      {'ps': ps_hosts,
       'worker': worker_hosts},
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_id)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    n_workers = len(worker_hosts)
    worker_id = int(FLAGS.task_id)
    dataset = mnist_data.load_mnist(worker_id=worker_id, n_workers=n_workers)
    # Only the chief checks for or creates train_dir.
    if FLAGS.task_id == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    distributed_train.train(server.target, dataset.train, cluster_spec)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()
