#
# Basic script for working with ec2 and distributed tensorflow.
#

from __future__ import print_function
import sys
import threading
import Queue
import paramiko as pm
import boto3
import time
import json
import os
from scp import SCPClient

class Cfg(dict):

   def __getitem__(self, item):
       item = dict.__getitem__(self, item)
       if type(item) == type([]):
           return [x % self if type(x) == type("") else x for x in item]
       if type(item) == type(""):
           return item % self
       return item

cfg = Cfg({
    "name" : "Timeout",      # Unique name for this specific configuration
    "key_name": "MaxLamKeyPair",          # Necessary to ssh into created instances

    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 49,
    "n_ps" : 1,
    "n_evaluators" : 1,                   # Continually validates the model on the validation data
    "num_replicas_to_aggregate" : "50",

    "method" : "reserved",

    # Region speficiation
    "region" : "us-west-2",
    "availability_zone" : "us-west-2b",

    # Machine type - instance type configuration.
    "master_type" : "t2.large",
    "worker_type" : "t2.large",
    "ps_type" : "t2.large",
    "evaluator_type" : "t2.large",
    "image_id": "ami-2306ba43",

    # Launch specifications
    "spot_price" : ".12",                 # Has to be a string

    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "/Users/maxlam/Desktop/School/Fall2016/Research/DistributedSGD/DistributedSGD.pem",

    # NFS configuration
    # To set up these values, go to Services > ElasticFileSystem > Create new filesystem, and follow the directions.
    #"nfs_ip_address" : "172.31.3.173",         # us-west-2c
    #"nfs_ip_address" : "172.31.35.0",          # us-west-2a
    "nfs_ip_address" : "172.31.28.54",          # us-west-2b
    "nfs_mount_point" : "/home/ubuntu/inception_shared",       # NFS base dir
    "base_out_dir" : "%(nfs_mount_point)s/%(name)s", # Master writes checkpoints to this directory. Outfiles are written to this directory.

    "setup_commands" :
    [
        "sudo rm -rf %(base_out_dir)s",
        "mkdir %(base_out_dir)s",
    ],

    # Command specification
    # Master pre commands are run only by the master
    "master_pre_commands" :
    [
        "cd DistributedMNIST",
        "git fetch && git reset --hard origin/master",
    ],

    # Pre commands are run on every machine before the actual training.
    "pre_commands" :
    [
        "cd DistributedMNIST",
        "git fetch && git reset --hard origin/master",
    ],

    # Model configuration
    "batch_size" : "128",
    "initial_learning_rate" : ".001",
    "learning_rate_decay_factor" : ".98",
    "num_epochs_per_decay" : "1.0",

    # Train command specifies how the ps/workers execute tensorflow.
    # PS_HOSTS - special string replaced with actual list of ps hosts.
    # TASK_ID - special string replaced with actual task index.
    # JOB_NAME - special string replaced with actual job name.
    # WORKER_HOSTS - special string replaced with actual list of worker hosts
    # ROLE_ID - special string replaced with machine's identity (E.G: master, worker0, worker1, ps, etc)
    # %(...)s - Inserts self referential string value.
    "train_commands" :
    [
        "python src/mnist_distributed_train.py "
        "--batch_size=%(batch_size)s "
        "--initial_learning_rate=%(initial_learning_rate)s "
        "--learning_rate_decay_factor=%(learning_rate_decay_factor)s "
        "--num_epochs_per_decay=%(num_epochs_per_decay)s "
        "--train_dir=%(base_out_dir)s/train_dir "
        "--worker_hosts='WORKER_HOSTS' "
        "--ps_hosts='PS_HOSTS' "
        "--task_id=TASK_ID "
        "--timeline_logging=false "
        "--interval_method=false "
        "--worker_times_cdf_method=false "
        "--interval_ms=1200 "
        "--num_replicas_to_aggregate=%(num_replicas_to_aggregate)s "
        "--job_name=JOB_NAME > %(base_out_dir)s/out_ROLE_ID 2>&1 &"
    ],

    # Commands to run on the evaluator
    "evaluate_commands" :
    [
        # Sleep a bit
        "sleep 30",

        # Evaluation command
        "python src/mnist_eval.py "
        "--eval_dir=%(base_out_dir)s/eval_dir "
        "--checkpoint_dir=%(base_out_dir)s/train_dir "
        "> %(base_out_dir)s/out_evaluator 2>&1 &",

        # Tensorboard command
        "python /usr/local/lib/python2.7/dist-packages/tensorflow/tensorboard/tensorboard.py "
        " --logdir=%(base_out_dir)s/train_dir/ "
        #" --logdir=%(base_out_dir)s/eval_dir/ "
        "> %(base_out_dir)s/out_evaluator_tensorboard 2>&1 &"
    ],
})

def tf_ec2_run(argv, configuration):

    client = boto3.client("ec2", region_name=configuration["region"])
    ec2 = boto3.resource("ec2", region_name=configuration["region"])

    def sleep_a_bit():
        time.sleep(5)

    def summarize_instances(instances):
        instance_type_to_instance_map = {}
        for instance in sorted(instances, key=lambda x:x.id):
            typ = instance.instance_type
            if typ not in instance_type_to_instance_map:
                instance_type_to_instance_map[typ] = []
            instance_type_to_instance_map[typ].append(instance)

        for k,v in instance_type_to_instance_map.items():
            print("%s - %d running" % (k, len(v)))

        return instance_type_to_instance_map

    def summarize_idle_instances(argv):
        print("Idle instances: (Idle = not running tensorflow)")
        summarize_instances(get_idle_instances())

    def summarize_running_instances(argv):
        print("Running instances: ")
        summarize_instances(ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}]))

    # Terminate all request.
    def terminate_all_requests():
         spot_requests = client.describe_spot_instance_requests()
         spot_request_ids = []
         for spot_request in spot_requests["SpotInstanceRequests"]:
            if spot_request["State"] != "cancelled" and spot_request["LaunchSpecification"]["KeyName"] == configuration["key_name"]:
               spot_request_id = spot_request["SpotInstanceRequestId"]
               spot_request_ids.append(spot_request_id)

         if len(spot_request_ids) != 0:
             print("Terminating spot requests: %s" % " ".join([str(x) for x in spot_request_ids]))
             client.cancel_spot_instance_requests(SpotInstanceRequestIds=spot_request_ids)

         # Wait until all are cancelled.
         # TODO: Use waiter class
         done = False
         while not done:
             print("Waiting for all spot requests to be terminated...")
             done = True
             spot_requests = client.describe_spot_instance_requests()
             states = [x["State"] for x in spot_requests["SpotInstanceRequests"] if x["LaunchSpecification"]["KeyName"] == configuration["key_name"]]
             for state in states:
                 if state != "cancelled":
                     done = False
             sleep_a_bit()

    # Terminate all instances in the configuration
    # Note: all_instances = ec2.instances.all() to get all intances
    def terminate_all_instances():
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        all_instance_ids = [x.id for x in live_instances]
        print([x.id for x in live_instances])
        if len(all_instance_ids) != 0:
            print("Terminating instances: %s" % (" ".join([str(x) for x in all_instance_ids])))
            client.terminate_instances(InstanceIds=all_instance_ids)

            # Wait until all are terminated
            # TODO: Use waiter class
            done = False
            while not done:
                print("Waiting for all instances to be terminated...")
                done = True
                instances = ec2.instances.all()
                for instance in instances:
                    if instance.state == "active":
                        done = False
                sleep_a_bit()

    # Launch instances as specified in the configuration.
    def launch_instances():
       method = "spot"
       if "method" in configuration.keys():
          method = configuration["method"]
       worker_instance_type, worker_count = configuration["worker_type"], configuration["n_workers"]
       master_instance_type, master_count = configuration["master_type"], configuration["n_masters"]
       ps_instance_type, ps_count = configuration["ps_type"], configuration["n_ps"]
       evaluator_instance_type, evaluator_count = configuration["evaluator_type"], configuration["n_evaluators"]
       specs = [(worker_instance_type, worker_count),
                (master_instance_type, master_count),
                (ps_instance_type, ps_count),
                (evaluator_instance_type, evaluator_count)]
       for (instance_type, count) in specs:
          launch_specs = {"KeyName" : configuration["key_name"],
                          "ImageId" : configuration["image_id"],
                          "InstanceType" : instance_type,
                          "Placement" : {"AvailabilityZone":configuration["availability_zone"]},
                          "SecurityGroups": ["default"]}
          if method == "spot":
             # TODO: EBS optimized? (Will incur extra hourly cost)
             client.request_spot_instances(InstanceCount=count,
                                           LaunchSpecification=launch_specs,
                                           SpotPrice=configuration["spot_price"])
          elif method == "reserved":
             client.run_instances(ImageId=launch_specs["ImageId"],
                                  MinCount=count,
                                  MaxCount=count,
                                  KeyName=launch_specs["KeyName"],
                                  InstanceType=launch_specs["InstanceType"],
                                  Placement=launch_specs["Placement"],
                                  SecurityGroups=launch_specs["SecurityGroups"])
          else:
             print("Unknown method: %s" % method)
             sys.exit(-1)


    # TODO: use waiter class?
    def wait_until_running_instances_initialized():
        done = False
        while not done:
            print("Waiting for instances to be initialized...")
            done = True
            live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
            ids = [x.id for x in live_instances]
            resps_list = [client.describe_instance_status(InstanceIds=ids[i:i+50]) for i in range(0, len(ids), 50)]
            statuses = []
            for resp in resps_list:
               statuses += [x["InstanceStatus"]["Status"] for x in resp["InstanceStatuses"]]
            #resps = client.describe_instance_status(InstanceIds=ids)
            #for resp in resps["InstanceStatuses"]:
            #    if resp["InstanceStatus"]["Status"] != "ok":
            #        done = False
            print(statuses)
            done = statuses.count("ok") == len(statuses)
            if len(ids) <= 0:
                done = False
            sleep_a_bit()

    # Waits until status requests are all fulfilled.
    # Prints out status of request in between time waits.
    # TODO: Use waiter class
    def wait_until_instance_request_status_fulfilled():
         requests_fulfilled = False
         n_active_or_open = 0
         while not requests_fulfilled or n_active_or_open == 0:
             requests_fulfilled = True
             statuses = client.describe_spot_instance_requests()
             print("InstanceRequestId, InstanceType, SpotPrice, State - Status : StatusMessage")
             print("-------------------------------------------")
             n_active_or_open = 0
             for instance_request in statuses["SpotInstanceRequests"]:
                 if instance_request["LaunchSpecification"]["KeyName"] != configuration["key_name"]:
                    continue
                 sid = instance_request["SpotInstanceRequestId"]
                 machine_type = instance_request["LaunchSpecification"]["InstanceType"]
                 price = instance_request["SpotPrice"]
                 state = instance_request["State"]
                 status, status_string = instance_request["Status"]["Code"], instance_request["Status"]["Message"]
                 if state == "active" or state == "open":
                     n_active_or_open += 1
                     print("%s, %s, %s, %s - %s : %s" % (sid, machine_type, price, state, status, status_string))
                     if state != "active":
                         requests_fulfilled = False
             print("-------------------------------------------")
             sleep_a_bit()

    # Create a client to the instance
    def connect_client(instance):
        client = pm.SSHClient()
        host = instance.public_ip_address
        client.set_missing_host_key_policy(pm.AutoAddPolicy())
        client.connect(host, username=configuration["ssh_username"], key_filename=configuration["path_to_keyfile"])
        return client

    # Takes a list of commands (E.G: ["ls", "cd models"]
    # and executes command on instance, returning the stdout.
    # Executes everything in one session, and returns all output from all the commands.
    def run_ssh_commands(instance, commands):
        done = False
        while not done:
           try:
              print("Instance %s, Running ssh commands:\n%s" % (instance.public_ip_address, "\n".join(commands)))

              # Always need to exit
              commands.append("exit")

              # Set up ssh client
              client = connect_client(instance)

              # Clear the stdout from ssh'ing in
              # For each command perform command and read stdout
              commandstring = "\n".join(commands)
              stdin, stdout, stderr = client.exec_command(commandstring)
              output = stdout.read()

              # Close down
              stdout.close()
              stdin.close()
              client.close()
              done = True
           except:
              done = False
        return output

    def run_ssh_commands_parallel(instance, commands, q):
        output = run_ssh_commands(instance, commands)
        q.put((instance, output))

    # Checks whether instance is idle. Assumed that instance is up and running.
    # An instance is idle if it is not running tensorflow...
    # Returns a tuple of (instance, is_instance_idle). We return a tuple for multithreading ease.
    def is_instance_idle(q, instance):
        python_processes = run_ssh_commands(instance, ["ps aux | grep python"])
        q.put((instance, not "ps_hosts" in python_processes and not "ps_workers" in python_processes))

    # Idle instances are running instances that are not running the inception model.
    # We check whether an instance is running the inception model by ssh'ing into a running machine,
    # and checking whether python is running.
    def get_idle_instances():
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        threads = []
        q = Queue.Queue()

        # Run commands in parallel, writing to the queue
        for instance in live_instances:
            t = threading.Thread(target=is_instance_idle, args=(q, instance))
            t.daemon = True
            t.start()
            threads.append(t)

        # Wait for threads to finish
        for thread in threads:
            thread.join()

        # Collect idle instances
        idle_instances = []
        while not q.empty():
            instance, is_idle = q.get()
            if is_idle:
                idle_instances.append(instance)

        return idle_instances

    def get_instance_requirements():
        # Get the requirements given the specification of worker/master/etc machine types
        worker_instance_type, worker_count = configuration["worker_type"], configuration["n_workers"]
        master_instance_type, master_count = configuration["master_type"], configuration["n_masters"]
        ps_instance_type, ps_count = configuration["ps_type"], configuration["n_ps"]
        evaluator_instance_type, evaluator_count = configuration["evaluator_type"], configuration["n_evaluators"]
        specs = [(worker_instance_type, worker_count),
                 (master_instance_type, master_count),
                 (ps_instance_type, ps_count),
                 (evaluator_instance_type, evaluator_count)]
        reqs = {}
        for (type_needed, count_needed) in specs:
            if type_needed not in reqs:
                reqs[type_needed] = 0
            reqs[type_needed] += count_needed
        return reqs

    # Returns whether the idle instances satisfy the specs of the configuration.
    def check_idle_instances_satisfy_configuration():
        # Create a map of instance types to instances of that type
        idle_instances = get_idle_instances()
        instance_type_to_instance_map = summarize_instances(idle_instances)

        # Get instance requirements
        reqs = get_instance_requirements()

        # Check the requirements are satisfied.
        print("Checking whether # of running instances satisfies the configuration...")
        for k,v in instance_type_to_instance_map.items():
            n_required = 0 if k not in reqs else reqs[k]
            print("%s - %d running vs %d required" % (k,len(v),n_required))
            if len(v) < n_required:
                print("Error, running instances failed to satisfy configuration requirements")
                sys.exit(0)
        print("Success, running instances satisfy configuration requirement")

    def shut_everything_down(argv):
        terminate_all_requests()
        terminate_all_instances()

    # Main method to run tf commands on a set of idle instances.
    def run_tf(argv, batch_size=128, port=1234):

        assert(configuration["n_masters"] == 1)

        # Check idle instances satisfy configs
        check_idle_instances_satisfy_configuration()

        # Get idle instances
        idle_instances = get_idle_instances()

        # Clear the nfs
        #instances_string = ",".join([x.instance_id for x in idle_instances])
        #clear_outdir_argv = ["python", "inception_ec2.py", instances_string, "sudo rm -rf %s" % configuration["base_out_dir"]]
        #run_command(clear_outdir_argv, quiet=True)
        #make_outdir_argv = ["python", "inception_ec2.py", instances_string, "mkdir %s" % configuration["base_out_dir"]]
        #run_command(make_outdir_argv, quiet=True)

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]},
            "ps" : {"instance_type" : configuration["ps_type"],
                    "n_required" : configuration["n_ps"]},
            "evaluator" : {"instance_type" : configuration["evaluator_type"],
                           "n_required" : configuration["n_evaluators"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : [],
            "ps" : [],
            "evaluator" : []
        }
        for role, requirement in sorted(specs.items(), key=lambda x:x[0]):
            instance_type_for_role = requirement["instance_type"]
            n_instances_needed = requirement["n_required"]
            instances_to_assign, rest = instance_type_to_instance_map[instance_type_for_role][:n_instances_needed], instance_type_to_instance_map[instance_type_for_role][n_instances_needed:]
            instance_type_to_instance_map[instance_type_for_role] = rest
            machine_assignments[role] = instances_to_assign

        # Construct the host strings necessary for running the inception command.
        # Note we use private ip addresses to avoid EC2 transfer costs.
        worker_host_string = ",".join([x.private_ip_address+":"+str(port) for x in machine_assignments["master"] + machine_assignments["worker"]])
        ps_host_string = ",".join([x.private_ip_address+":"+str(port) for x in machine_assignments["ps"]])

        # Create a map of command&machine assignments
        command_machine_assignments = {}
        setup_machine_assignments = {}

        # Construct the master command
        command_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["master_pre_commands"])}
        setup_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["setup_commands"])}
        for command_string in configuration["train_commands"]:
            command_machine_assignments["master"]["commands"].append(command_string.replace("PS_HOSTS", ps_host_string).replace("TASK_ID", "0").replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", "master"))

        # Construct the worker commands
        for worker_id, instance in enumerate(machine_assignments["worker"]):
            name = "worker_%d" % worker_id
            command_machine_assignments[name] = {"instance" : instance,
                                                 "commands" : list(configuration["pre_commands"])}
            for command_string in configuration["train_commands"]:
                command_machine_assignments[name]["commands"].append(command_string.replace("PS_HOSTS", ps_host_string).replace("TASK_ID", "%d" % (worker_id+1)).replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))

        # Construct ps commands
        for ps_id, instance in enumerate(machine_assignments["ps"]):
            name = "ps_%d" % ps_id
            command_machine_assignments[name] = {"instance" : instance,
                                                 "commands" : list(configuration["pre_commands"])}
            for command_string in configuration["train_commands"]:
                command_machine_assignments[name]["commands"].append(command_string.replace("PS_HOSTS", ps_host_string).replace("TASK_ID", "%d" % ps_id).replace("JOB_NAME", "ps").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))

        # The evaluator requires a special command to continually evaluate accuracy on validation data.
        # We also launch the tensorboard on it.
        assert(len(machine_assignments["evaluator"]) == 1)
        command_machine_assignments["evaluator"] = {"instance" : machine_assignments["evaluator"][0],
                                                    "commands" : list(configuration["pre_commands"]) + list(configuration["evaluate_commands"])}

        # Run the commands via ssh in parallel
        threads = []
        q = Queue.Queue()

        for name, command_and_machine in setup_machine_assignments.items():
            instance = command_and_machine["instance"]
            commands = command_and_machine["commands"]
            print("-----------------------")
            print("Pre Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)

        # Wait until commands are all finished
        for t in threads:
            t.join()

        threads = []
        q = Queue.Queue()

        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            commands = command_and_machine["commands"]
            print("-----------------------")
            print("Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)

        # Wait until commands are all finished
        for t in threads:
            t.join()

        # Print the output
        while not q.empty():
            instance, output = q.get()
            print(instance.public_ip_address)
            print(output)

        # Debug print
        instances = []
        print("\n--------------------------------------------------\n")
        print("Machine assignments:")
        print("------------------------")
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            instances.append(instance)
            commands = command_and_machine["commands"]
            ssh_command = "ssh -i %s %s@%s" % (configuration["path_to_keyfile"], configuration["ssh_username"], instance.public_ip_address)
            print("%s - %s" % (name, instance.instance_id))
            print("To ssh: %s" % ssh_command)
            print("------------------------")

        # Print out list of instance ids (which will be useful in selctively stopping inception
        # for given instances.
        instance_cluster_string = ",".join([x.instance_id for x in instances])
        print("\nInstances cluster string: %s" % instance_cluster_string)

        # Print out the id of the configuration file
        cluster_save = {
            "configuration" : configuration,
            "name" : configuration["name"],
            "command_machine_assignments" : command_machine_assignments,
            "cluster_string" : instance_cluster_string
        }

        return cluster_save

    def kill_python(argv):
        if len(argv) != 3:
            print("Usage: python inception_ec2.py kill_python instance_id1,instance_id2,id3...")
            sys.exit(0)
        cluster_instance_string = argv[2]
        instance_ids_to_shutdown = cluster_instance_string.split(",")

        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
        threads = []
        q = Queue.Queue()
        for instance in live_instances:
            if instance.instance_id in instance_ids_to_shutdown:
                commands = ["sudo pkill -9 python"]
                t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
                t.start()
                threads.append(t)
        for thread in threads:
            thread.join()
        summarize_idle_instances(None)

    def kill_all_python(argv):
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']},  {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        threads = []
        q = Queue.Queue()
        for instance in live_instances:
            commands = ["sudo pkill -9 python"]
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
        summarize_idle_instances(None)

    def download_file(argv):
       if len(argv) != 5:
          print("Usage python inception_ec2.py download_file instances_cluster_string path_to_file dir_to_save")
          sys.exit(0)

       # Get instances of the cluster
       cluster_instance_string = argv[2]
       instance_ids = cluster_instance_string.split(",")

       filepath = argv[3]

       # Create the outpath if it does not exist
       outpath = argv[4] + "/"
       if not os.path.exists(outpath):
          os.makedirs(outpath)

       # Get a random running instance (does not have to be idle, since
       # might want to download while everything machine is not idle)
       running_instances = [x for x in ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])]
       if len(running_instances) == 0:
          print("Error, no running instances")
          sys.exit(0)
       selected_instance = None
       for instance in running_instances:
          if instance.instance_id in instance_ids:
             selected_instance = instance
       if selected_instance == None:
          print("Error, no instance in instance cluster: %s" % cluster_instance_string)
          sys.exit(0)

       # For the selected instance, ssh and compress the directory
       file_to_download = configuration["base_out_dir"] + "/" + filepath
       name = configuration["name"] + "_data_" + filepath
       copy_command = "cp -r %s ./%s" % (file_to_download, name)
       run_ssh_commands(instance, [copy_command])

       # SCP the data over to the local machine
       client = connect_client(selected_instance)
       scp = SCPClient(client.get_transport())
       local_path = outpath + name
       print("SCP %s to %s" % (name, local_path))
       scp.get("%s" % name, local_path=local_path)
       scp.close()
       client.close()

       return local_path

    def download_outdir(argv):
        if len(argv) != 4:
            print("Usage python inception_ec2.py download_outdir instances_cluster_string path_to_save")
            sys.exit(0)

        # Get instances of the cluster
        cluster_instance_string = argv[2]
        instance_ids = cluster_instance_string.split(",")

        # Create the outpath if it does not exist
        outpath = argv[3] + "/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Get a random running instance (does not have to be idle, since
        # might want to download while everything machine is not idle)
        running_instances = [x for x in ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])]
        if len(running_instances) == 0:
            print("Error, no running instances")
            sys.exit(0)
        selected_instance = None
        for instance in running_instances:
            if instance.instance_id in instance_ids:
                selected_instance = instance
        if selected_instance == None:
            print("Error, no instance in instance cluster: %s" % cluster_instance_string)
            sys.exit(0)

        # For the selected instance, ssh and compress the directory
        directory_to_download = configuration["base_out_dir"]
        name = configuration["name"] + "_data"
        copy_command = "cp -r %s ./%s" % (directory_to_download, name)
        compress_command = "tar -czf %s.tar.gz %s" % (name, name)
        run_ssh_commands(instance, [copy_command, compress_command])

        # SCP the data over to the local machine
        client = connect_client(selected_instance)
        scp = SCPClient(client.get_transport())
        local_path = outpath + name + ".tar.gz"
        print("SCP %s.tar.gz to %s" % (name, local_path))
        scp.get("%s.tar.gz" % name, local_path=local_path)
        scp.close()
        client.close()

    def run_command(argv, quiet=False):
        if len(argv) != 4:
            print("Usage: python inception_ec2.py run_command instance_id1,instance_id2,id3... command")
            sys.exit(0)
        cluster_instance_string = argv[2]
        command = argv[3]
        instance_ids_to_run_command = cluster_instance_string.split(",")

        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        threads = []
        q = Queue.Queue()
        for instance in live_instances:
            if instance.instance_id in instance_ids_to_run_command:
                commands = [command]
                t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
                t.start()
                threads.append(t)
        for thread in threads:
            thread.join()

        while not q.empty():
            instance, output = q.get()
            if not quiet:
                print(instance, output)

    # Setup nfs on all instances
    def setup_nfs():
        print("Clearing previous nfs file system...")
        live_instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}, {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
        live_instances_string = ",".join([x.instance_id for x in live_instances])
        rm_command = "sudo rm -rf %s" % configuration["nfs_mount_point"]
        argv = ["python", "inception_ec2.py", live_instances_string, rm_command]
        run_command(argv, quiet=True)

        print("Installing nfs on all running instances...")
        update_command = "sudo apt-get -y update"
        install_nfs_command = "sudo apt-get -y install nfs-common"
        create_mount_command = "mkdir %s" % configuration["nfs_mount_point"]
        setup_nfs_command = "sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ %s" % (configuration["nfs_ip_address"], configuration["nfs_mount_point"])
        reduce_permissions_command = "sudo chmod 777 %s " % configuration["nfs_mount_point"]
        command = update_command + " && " + install_nfs_command + " && " + create_mount_command + " && " + setup_nfs_command + " && " + reduce_permissions_command

        # pretty hackish
        argv = ["python", "inception_ec2.py", live_instances_string, command]
        run_command(argv, quiet=True)

    # Launch instances as specified by the configuration.
    # We also want a shared filesystem to write model checkpoints.
    # For simplicity we will have the user specify the filesystem via the config.
    def launch(argv):
        method = "spot"
        if "method" in configuration:
           method = configuration["method"]
        launch_instances()
        if method == "spot":
           wait_until_instance_request_status_fulfilled()
        wait_until_running_instances_initialized()
        setup_nfs()

    def clean_launch_and_run(argv):
        # 1. Kills all instances in region
        # 2. Kills all requests in region
        # 3. Launches requests
        # 5. Waits until launch requests have all been satisfied,
        #    printing status outputs in the meanwhile
        # 4. Checks that configuration has been satisfied
        # 5. Runs inception
        shut_everything_down(None)
        launch(None)
        return run_tf(None)

    def help(hmap):
        print("Usage: python inception_ec2.py [command]")
        print("Commands:")
        for k,v in hmap.items():
            print("%s - %s" % (k,v))

    ##############################
    # tf_ec2 main starting point #
    ##############################

    command_map = {
        "launch" : launch,
        "clean_launch_and_run" : clean_launch_and_run,
        "shutdown" : shut_everything_down,
        "run_tf" : run_tf,
        "kill_all_python" : kill_all_python,
        "list_idle_instances" : summarize_idle_instances,
        "list_running_instances" : summarize_running_instances,
        "kill_python" : kill_python,
        "run_command" : run_command,
        "download_outdir" : download_outdir,
        "download_file" : download_file,
    }
    help_map = {
        "launch" : "Launch instances",
        "clean_launch_and_run" : "Shut everything down, launch instances, wait until requests fulfilled, check that configuration is fulfilled, and launch and run inception.",
        "shutdown" : "Shut everything down by cancelling all instance requests, and terminating all instances.",
        "list_idle_instances" : "Lists all idle instances. Idle instances are running instances not running tensorflow.",
        "list_running_instances" : "Lists all running instances.",
        "run_tf" : "Runs inception on idle instances.",
        "kill_all_python" : "Kills python running inception training on ALL instances.",
        "kill_python" : "Kills python running inception on instances indicated by instance id string separated by ',' (no spaces).",
        "run_command" : "Runs given command on instances selcted by instance id string, separated by ','.",
        "download_outdir" : "Downloads base_out_dir as specified in the configuration. Used for pulling checkpoint files and saved models.",
        "download_file" : "Downloads base_out_dir/filepath as specified in the configuration."
    }

    if len(argv) < 2:
        help(help_map)
        sys.exit(0)

    command = argv[1]
    return command_map[command](argv)

if __name__ == "__main__":
    print(cfg)
    tf_ec2_run(sys.argv, cfg)
