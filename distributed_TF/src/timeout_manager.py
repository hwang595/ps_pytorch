import signal
import sys
import os
import time

import tensorflow as tf
import numpy as np
from twisted.spread import pb
from twisted.internet import reactor
from threading import Thread, Timer
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.protocols.policies import TimeoutMixin
from twisted.internet.protocol import Protocol

##################
# RPC procedures #
##################
class TimeoutServer(pb.Root):
  def __init__(self, tf_flags, sess):
    self.sess = sess
    self.tf_flags = tf_flags
    self.worker_id = self.tf_flags.task_id
    self.n_total_workers = len(self.tf_flags.worker_hosts.split(","))
    self.iteration_track = [0] * self.n_total_workers
    self.n_to_collect = self.tf_flags.num_replicas_to_aggregate
    self.ready_to_start = False
    self.iterations_killed = set()
    tf.logging.info("Worker %d: starting status server..." % self.tf_flags.task_id)

    # Statistics tracking
    self.worker_dequeue_times = [{}] * self.n_total_workers
    self.worker_finished_computing_gradients_times = [{}] * self.n_total_workers
    self.compute_times = []
    self.iteration_start_times = {}
    self.ITERATION_START_TRACKING = 10
    self.ITERATION_END_TRACKING = 500

  def remote_parameters_updated(self, step):
    tf.logging.info("Parameters have been updated on step %d.." % step)
    tf.logging.info("Start killing at time %f" % time.time())
    try:
      if self.worker_id != 0:
        #self.sess.kill()
        pass
    except:
      tf.logging.info("Unexpected error:", sys.exc_info()[0])

  def remote_worker_dequeued_token(self, worker_id, iteration):
    tf.logging.info("Worker %d dequeued token on iteration %d - %d" % (worker_id, iteration, time.time()))
    cur_time = time.time()
    self.worker_dequeue_times[worker_id][iteration] = cur_time
    if iteration not in self.iteration_start_times:
      self.iteration_start_times[iteration] = cur_time

  def remote_worker_finished_computing_gradients(self, worker_id, iteration):
    tf.logging.info("Worker %d finished computing gradients on iteration %d - %d" % (worker_id, iteration, time.time()))
    cur_time = time.time()
    self.worker_finished_computing_gradients_times[worker_id][iteration] = cur_time

    elapsed_time = self.worker_finished_computing_gradients_times[worker_id][iteration] - self.worker_dequeue_times[worker_id][iteration]
    self.compute_times.append((worker_id, elapsed_time, iteration))

    if iteration > self.ITERATION_START_TRACKING and worker_id == 0:
      elapsed_times = sorted([(x[1],x[0],x[2]) for x in self.compute_times if x[2] > self.ITERATION_START_TRACKING], key=lambda x: x[0])
      selected_iteration_start_times = [x for i,x in self.iteration_start_times.items() if i > self.ITERATION_START_TRACKING]
      iteration_times = [selected_iteration_start_times[i+1] - selected_iteration_start_times[i] for i in range(len(selected_iteration_start_times)-1)]

      if iteration % 50 == 0 or iteration == self.ITERATION_END_TRACKING:
        tf.logging.info("ELAPSED TIMES %s" % str(elapsed_times))
        tf.logging.info("ITERATION TIMES %s" % str(iteration_times))

  def remote_notify_ready_to_start(self):
    tf.logging.info("Server ready to start!")
    self.ready_to_start = True

  def remote_is_ready_to_start(self):
    return (self.worker_id, self.ready_to_start)

class RetryTimeoutProtocol(Protocol, TimeoutMixin):

    def __init__(self, factory):
        self.factory = factory

    def connectionMade(self):
        self.setTimeout(30)

    def dataReceived(self, data):
        self.resetTimeout()

    def timeoutConnection(self):
        tf.logging.info("Timed out...")
        self.transport.abortConnection()

class TimeoutReconnectClientFactory(pb.PBClientFactory, ReconnectingClientFactory):

    def startedConnecting(self, connector):
        tf.logging.info('Started to connect.')

    def clientConnectionLost(self, connector, reason):
        tf.logging.info('Lost connection.  Reason: %s' % str(reason))

    def clientConnectionFailed(self, connector, reason):
        tf.logging.info('Connection failed. Reason: %s' % str(reason))
        ReconnectingClientFactory.clientConnectionFailed(self, connector, reason)
        #connector.connect()

class TimeoutClient():

  def __init__(self, tf_flags):
    self.tf_flags = tf_flags
    self.worker_id = self.tf_flags.task_id
    hosts = self.tf_flags.worker_hosts.split(",")
    hosts = [x.split(":")[0] for x in hosts]
    self.hosts = hosts
    self.self_perspective = None
    self.perspectives = []
    self.ready = False
    self.servers_ready = set([])

    for i, host in enumerate(hosts):
      #factory = pb.PBClientFactory()
      factory = TimeoutReconnectClientFactory()
      tf.logging.info("Connecting to %s:%d" % (host, self.tf_flags.rpc_port))
      reactor.connectTCP(host, self.tf_flags.rpc_port, factory)
      if i == self.worker_id:
        factory.getRootObject().addCallback(self.connected_self)
        #factory.getRootObject().addCallbacks(self.connected_self, self.connect_failure, errbackArgs=[host], errbackKeywords=[])
      else:
        factory.getRootObject().addCallback(self.connected)
        #factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=[host], errbackKeywords=[])

  def broadcast_parameters_updated(self, step):
    for persp in self.perspectives:
      persp.callRemote("parameters_updated", step)

  def broadcast_worker_dequeued_token(self, iteration):
    for persp in self.perspectives:
      persp.callRemote("worker_dequeued_token", self.worker_id, iteration)

  def broadcast_worker_finished_computing_gradients(self, iteration):
    for persp in self.perspectives:
      persp.callRemote("worker_finished_computing_gradients", self.worker_id, iteration)

  def server_ready_to_start(self, *args):
    wid, ready = args[0]
    if ready:
      tf.logging.info("Worker %d is ready to begin..." % wid)
      self.servers_ready.add(wid)

  def check_ready_to_start(self):
    tf.logging.info("Checking ready to start: %d" % len(self.perspectives))
    for persp in self.perspectives:
      persp.callRemote("is_ready_to_start").addCallbacks(self.server_ready_to_start, self.fail)

  def ready_to_start(self):
    tf.logging.info(self.servers_ready)
    tf.logging.info("Num servers ready: %d vs %d" % (len(self.servers_ready), len(self.hosts)))
    return self.ready and len(self.servers_ready) == len(self.hosts)

  def signal_server_ready(self):
    tf.logging.info("Signaling ready to self's server")
    self.self_perspective.callRemote("notify_ready_to_start").addCallbacks(self.success, self.fail)

  def connected(self, perspective):
    self.perspectives.append(perspective)
    tf.logging.info(str(perspective))
    tf.logging.info("Connected!")
    tf.logging.info("%d %d" % (len(self.hosts), len(self.perspectives)))
    self.ready = (len(self.hosts) == len(self.perspectives))
    if self.ready:
      tf.logging.info("Ready!")
      self.signal_server_ready()
    else:
      tf.logging.info("%d of %d" % (len(self.perspectives), len(self.hosts)))

  def connected_self(self, perspective):
    self.self_perspective = perspective
    self.connected(perspective)

  def success(self, result):
    #tf.logging.info("Success!")
    pass

  def fail(self, _):
    tf.logging.info("Fail")
    tf.logging.info(_)

  def connect_failure(self, *args, **kwargs):
    tf.logging.info("RPC error, something failed: ")
    #time.sleep(1)
    #host = "".join(args[1:])
    #factory = pb.PBClientFactory()
    #tf.logging.info("Trying reconnecting to %s:%d" % (host, self.tf_flags.rpc_port))
    #reactor.connectTCP(host, self.tf_flags.rpc_port, factory)
    #factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=(host))

# Separate manager process to oversee training on workers.
def launch_manager(sess, tf_flags):
  # Launch a separate thread in the background that checks whether the
  # machine is a straggler.
  timeout_server = TimeoutServer(tf_flags, sess)
  rpc_server = pb.PBServerFactory(timeout_server)
  reactor.listenTCP(tf_flags.rpc_port, rpc_server)
  rpc_client = TimeoutClient(tf_flags)
  Thread(target=reactor.run, args=(False,)).start()

  while not rpc_client.ready_to_start():
    rpc_client.check_ready_to_start()
    time.sleep(1)

  return rpc_client, timeout_server,
