from __future__ import print_function
import sys
import threading
import Queue
import paramiko as pm
import boto3
import time
import json
import os


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
    "key_name": "HongyiScript",          # Necessary to ssh into created instances
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 3,
    "num_replicas_to_aggregate" : "8",
    "method" : "reserved",
    # Region speficiation
    "region" : "us-west-2",
    "availability_zone" : "us-west-2b",
    # Machine type - instance type configuration.
    "master_type" : "t2.large",
    "worker_type" : "t2.large",
    # please only use this AMI for pytorch
    "image_id": "ami-1036c268",
    # Launch specifications
    "spot_price" : "0.8",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "/home/hwang/My_Code/AWS/HongyiScript.pem",

    # NFS configuration
    # To set up these values, go to Services > ElasticFileSystem > Create new filesystem, and follow the directions.
    #"nfs_ip_address" : "172.31.3.173",         # us-west-2c
    #"nfs_ip_address" : "172.31.35.0",          # us-west-2a
    "nfs_ip_address" : "172.31.14.225",          # us-west-2b
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
    "base_out_dir" : "%(nfs_mount_point)s/%(name)s", # Master writes checkpoints to this directory. Outfiles are written to this directory.
    "setup_commands" :
    [
        # "sudo rm -rf %(base_out_dir)s",
        "mkdir %(base_out_dir)s",
    ],
    # Command specification
    # Master pre commands are run only by the master
    "master_pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
        "ls",
        # "cd distributed_tensorflow/DistributedResNet",
        # "git fetch && git reset --hard origin/master",
    ],
    # Pre commands are run on every machine before the actual training.
    "pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
    ],
    # Model configuration
    "batch_size" : "32",
    "max_steps" : "2000",
    "initial_learning_rate" : ".001",
    "learning_rate_decay_factor" : ".95",
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
        "echo ========= Start ==========="
    ],
})

def mxnet_ec2_run(argv, configuration):
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

        for type in instance_type_to_instance_map:
            print("Type\t", type)
            for instance in instance_type_to_instance_map[type]:
                print("instance\t", instance, "\t", instance.public_ip_address)
            print

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
       specs = [(worker_instance_type, worker_count),
                (master_instance_type, master_count)]
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
              print("Instance %s, Running ssh commands:\n%s\n" % (instance.public_ip_address, "\n".join(commands)))

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
           except Exception as e:
              done = False
              print(e.message)
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
        live_instances = ec2.instances.filter(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']},
                     {'Name': 'key-name', 'Values': [configuration["key_name"]]}])
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
        specs = [(worker_instance_type, worker_count),
                 (master_instance_type, master_count)]
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

    def run_mxnet_grid_search(argv, port=1334):
        # Check idle instances satisfy configs
        check_idle_instances_satisfy_configuration()

        # Get idle instances
        idle_instances = get_idle_instances()

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : []
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

        # Create a map of command&machine assignments
        command_machine_assignments = {}
        setup_machine_assignments = {}

        # Construct the master command
        command_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["master_pre_commands"])}
        # setup_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["setup_commands"])}
        for command_string in configuration["train_commands"]:
            command_machine_assignments["master"]["commands"].append(command_string.replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", "master"))
        print(command_machine_assignments)

        # Construct the worker commands
        for worker_id, instance in enumerate(machine_assignments["worker"]):
            name = "worker_%d" % worker_id
            command_machine_assignments[name] = {"instance" : instance,
                                                 "commands" : list(configuration["pre_commands"])}
            for command_string in configuration["train_commands"]:
                command_machine_assignments[name]["commands"].append(command_string.replace("TASK_ID", "%d" % (worker_id+1)).replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))

        print(command_machine_assignments)

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

        running_process = 0
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            neo_commands = "python train_cifar10.py --running_mode=grid_search --gpus=0 "\
                           "--running_process={} "\
                           "--batch-size={} "\
                           "--dir={}/grid_search> {}/grid_search/batch_size_{}/running_{}_process.out 2>&1 &".format(
                running_process,
                configuration['batch_size'],
                configuration['nfs_mount_point'],
                configuration['nfs_mount_point'],
                configuration['batch_size'],
                running_process)

            commands = command_and_machine["commands"]
            commands.append('mkdir {}/grid_search'.format(configuration['nfs_mount_point']))
            commands.append('mkdir {}/grid_search/batch_size_{}'.format(
                configuration['nfs_mount_point'],
                configuration['batch_size']))
            commands.append(neo_commands)

            print("-----------------------")
            print("Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
            running_process += 1

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


    def run_mxnet_loss_curve(argv, port=1334):
        # Check idle instances satisfy configs
        check_idle_instances_satisfy_configuration()

        # Get idle instances
        idle_instances = get_idle_instances()

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : []
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

        # Create a map of command&machine assignments
        command_machine_assignments = {}
        setup_machine_assignments = {}

        # Construct the master command
        command_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["master_pre_commands"])}
        # setup_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["setup_commands"])}
        for command_string in configuration["train_commands"]:
            command_machine_assignments["master"]["commands"].append(command_string.replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", "master"))
        print(command_machine_assignments)

        # Construct the worker commands
        for worker_id, instance in enumerate(machine_assignments["worker"]):
            name = "worker_%d" % worker_id
            command_machine_assignments[name] = {"instance" : instance,
                                                 "commands" : list(configuration["pre_commands"])}
            for command_string in configuration["train_commands"]:
                command_machine_assignments[name]["commands"].append(command_string.replace("TASK_ID", "%d" % (worker_id+1)).replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))

        print(command_machine_assignments)

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

        batch_size_list = [4, 32, 50, 100, 500, 1000]
        learning_rate_list = [0.046, 0.05, 0.068, 0.068, 0.048, 0.086]
        running_process = 0
        for name, command_and_machine in command_machine_assignments.items():
            instance = command_and_machine["instance"]
            neo_commands = "python train_cifar10.py --running_mode=training --gpus=0 "\
                           "--batch-size={} "\
                           "--lr={} "\
                           "--model-prefix={}/model_checkpoints/batch_size_{} "\
                           "--dir={}/loss_curve > "\
                           "{}/loss_curve/running_batch_size_{}.out 2>&1 &".format(
                batch_size_list[running_process],
                learning_rate_list[running_process],
                configuration['nfs_mount_point'],
                batch_size_list[running_process],
                configuration['nfs_mount_point'],
                configuration['nfs_mount_point'],
                batch_size_list[running_process])

            commands = command_and_machine["commands"]
            commands.append('mkdir {}/model_checkpoints/'.format(configuration['nfs_mount_point']))
            commands.append('mkdir {}/loss_curve'.format(configuration['nfs_mount_point']))
            commands.append(neo_commands)

            print("-----------------------")
            print("Command: %s\n" % " ".join(commands))
            t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
            t.start()
            threads.append(t)
            running_process += 1

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



    def get_hosts(argv, port=22):
        # Check idle instances satisfy configs
        check_idle_instances_satisfy_configuration()

        # Get idle instances
        idle_instances = get_idle_instances()

        # Assign instances for worker/ps/etc
        instance_type_to_instance_map = summarize_instances(idle_instances)
        specs = {
            "master" : {"instance_type" : configuration["master_type"],
                        "n_required" : configuration["n_masters"]},
            "worker" : {"instance_type" : configuration["worker_type"],
                        "n_required" : configuration["n_workers"]}
        }
        machine_assignments = {
            "master" : [],
            "worker" : []
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
        hosts_out = open('hosts', 'w')
        print('master ip ', machine_assignments['master'][0].public_ip_address)
        count = 0
        for instance in machine_assignments["master"] + machine_assignments["worker"]:
            count += 1
            print('{}\tdeeplearning-worker{}'.format(instance.private_ip_address, count), end='\n', file=hosts_out)
        hosts_out.flush()
        hosts_out.close()

        hosts_alias_out = open('hosts_alias', 'w')
        count = 0
        for _ in machine_assignments["master"] + machine_assignments["worker"]:
            count += 1
            print('deeplearning-worker{}'.format(count), end='\n', file=hosts_alias_out)
        hosts_alias_out.flush()
        hosts_alias_out.close()

        hosts_alias_out = open('hosts_address', 'w')
        count = 0
        for instance in machine_assignments["master"] + machine_assignments["worker"]:
            count += 1
            print('{}'.format(instance.private_ip_address), end='\n', file=hosts_alias_out)
        hosts_alias_out.flush()
        hosts_alias_out.close()

        # # Create a map of command&machine assignments
        # command_machine_assignments = {}
        # setup_machine_assignments = {}
        #
        # # Construct the master command
        # command_machine_assignments["master"] = {"instance" : machine_assignments["master"][0], "commands" : list(configuration["master_pre_commands"])}
        # for command_string in configuration["train_commands"]:
        #     command_machine_assignments["master"]["commands"].append(command_string.replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", "master"))
        # print(command_machine_assignments)
        #
        # # Construct the worker commands
        # for worker_id, instance in enumerate(machine_assignments["worker"]):
        #     name = "worker_%d" % worker_id
        #     command_machine_assignments[name] = {"instance" : instance,
        #                                          "commands" : list(configuration["pre_commands"])}
        #     for command_string in configuration["train_commands"]:
        #         command_machine_assignments[name]["commands"].append(command_string.replace("TASK_ID", "%d" % (worker_id+1)).replace("JOB_NAME", "worker").replace("WORKER_HOSTS", worker_host_string).replace("ROLE_ID", name))
        #
        # print(command_machine_assignments)
        #
        # # Run the commands via ssh in parallel
        # threads = []
        # q = Queue.Queue()
        #
        # for name, command_and_machine in setup_machine_assignments.items():
        #     instance = command_and_machine["instance"]
        #     commands = command_and_machine["commands"]
        #     print("-----------------------")
        #     print("Pre Command: %s\n" % " ".join(commands))
        #     t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
        #     t.start()
        #     threads.append(t)
        #
        # # Wait until commands are all finished
        # for t in threads:
        #     t.join()
        #
        # threads = []
        # q = Queue.Queue()
        #
        # batch_size_list = [4, 32, 50, 100, 500, 1000]
        # learning_rate_list = [0.046, 0.05, 0.068, 0.068, 0.048, 0.086]
        # running_process = 0
        # for name, command_and_machine in command_machine_assignments.items():
        #     instance = command_and_machine["instance"]
        #     neo_commands = "python train_cifar10.py --running_mode=training --gpus=0 "\
        #                    "--batch-size={} "\
        #                    "--lr={} "\
        #                    "--model-prefix={}/model_checkpoints/batch_size_{} "\
        #                    "--dir={}/loss_curve > "\
        #                    "{}/loss_curve/running_batch_size_{}.out 2>&1 &".format(
        #         batch_size_list[running_process],
        #         learning_rate_list[running_process],
        #         configuration['nfs_mount_point'],
        #         batch_size_list[running_process],
        #         configuration['nfs_mount_point'],
        #         configuration['nfs_mount_point'],
        #         batch_size_list[running_process])
        #
        #     commands = command_and_machine["commands"]
        #     commands.append('mkdir {}/model_checkpoints/'.format(configuration['nfs_mount_point']))
        #     commands.append('mkdir {}/loss_curve'.format(configuration['nfs_mount_point']))
        #     commands.append(neo_commands)
        #
        #     print("-----------------------")
        #     print("Command: %s\n" % " ".join(commands))
        #     t = threading.Thread(target=run_ssh_commands_parallel, args=(instance, commands, q))
        #     t.start()
        #     threads.append(t)
        #     running_process += 1
        #
        # # Wait until commands are all finished
        # for t in threads:
        #     t.join()
        #
        # # Print the output
        # while not q.empty():
        #     instance, output = q.get()
        #     print(instance.public_ip_address)
        #     print(output)
        #
        # # Debug print
        # instances = []
        # print("\n--------------------------------------------------\n")
        # print("Machine assignments:")
        # print("------------------------")
        # for name, command_and_machine in command_machine_assignments.items():
        #     instance = command_and_machine["instance"]
        #     instances.append(instance)
        #     commands = command_and_machine["commands"]
        #     ssh_command = "ssh -i %s %s@%s" % (configuration["path_to_keyfile"], configuration["ssh_username"], instance.public_ip_address)
        #     print("%s - %s" % (name, instance.instance_id))
        #     print("To ssh: %s" % ssh_command)
        #     print("------------------------")
        #
        # # Print out list of instance ids (which will be useful in selctively stopping inception
        # # for given instances.
        # instance_cluster_string = ",".join([x.instance_id for x in instances])
        # print("\nInstances cluster string: %s" % instance_cluster_string)
        #
        # # Print out the id of the configuration file
        # cluster_save = {
        #     "configuration" : configuration,
        #     "name" : configuration["name"],
        #     "command_machine_assignments" : command_machine_assignments,
        #     "cluster_string" : instance_cluster_string
        # }
        #
        # return cluster_save
        return

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
        # argv = ["python", "inception_ec2.py", live_instances_string]
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
        return

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
        print('setup nfs')
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
        return run_mxnet_grid_search(None)

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
        "run_mxnet_grid_search": run_mxnet_grid_search,
        "run_mxnet_loss_curve": run_mxnet_loss_curve,
        "get_hosts": get_hosts,
        "kill_all_python" : kill_all_python,
        "list_idle_instances" : summarize_idle_instances,
        "list_running_instances" : summarize_running_instances,
        "kill_python" : kill_python,
        "run_command" : run_command,
        "setup_nfs": setup_nfs,
    }
    help_map = {
        "launch" : "Launch instances",
        "clean_launch_and_run" : "Shut everything down, launch instances, wait until requests fulfilled, check that configuration is fulfilled, and launch and run inception.",
        "shutdown" : "Shut everything down by cancelling all instance requests, and terminating all instances.",
        "list_idle_instances" : "Lists all idle instances. Idle instances are running instances not running tensorflow.",
        "list_running_instances" : "Lists all running instances.",
        "run_mxnet_grid_search": "",
        "run_mxnet_loss_curve": "",
        "setup_nfs": "",
        "kill_all_python" : "Kills python running inception training on ALL instances.",
        "kill_python" : "Kills python running inception on instances indicated by instance id string separated by ',' (no spaces).",
        "run_command" : "Runs given command on instances selcted by instance id string, separated by ','.",
    }

    if len(argv) < 2:
        help(help_map)
        sys.exit(0)

    command = argv[1]
    return command_map[command](argv)

if __name__ == "__main__":
    print(cfg)
    mxnet_ec2_run(sys.argv, cfg)