from __future__ import print_function
import sys
import json
import time
import numpy as np
import re
import shutil
import os
import glob
from matplotlib import pyplot as plt
from tf_ec2 import tf_ec2_run, Cfg

def load_cfg_from_file(cfg_file):
    cfg_f = open(cfg_file, "r")
    return eval(cfg_f.read())

def shutdown_and_launch(cfg):
    shutdown_args = "tools/tf_ec2.py shutdown"
    tf_ec2_run(shutdown_args.split(), cfg)

    launch_args = "tools/tf_ec2.py launch"
    tf_ec2_run(launch_args.split(), cfg)

def check_if_reached_iters(cluster_string, n_iters, cfg, master_file_name="out_master", outdir="/tmp/"):
    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, master_file_name, outdir)
    fname = tf_ec2_run(download_evaluator_file_args.split(), cfg)
    f = open(fname, "r")
    cur_iteration = 0
    for line in f:
        m = re.match(".*step ([0-9]*),.*", line)
        if m:
            cur_iteration = max(cur_iteration, int(m.group(1)))
    print("Currently on iteration %d" % cur_iteration)
    return cur_iteration > n_iters

def run_tf_and_download_files(n_iters, cfg, evaluator_file_name="out_evaluator", master_file_name="out_master", outdir="result_dir"):

    kill_args = "tools/tf_ec2.py kill_all_python"
    tf_ec2_run(kill_args.split(), cfg)
    time.sleep(10)

    run_args = "tools/tf_ec2.py run_tf"
    cluster_specs = tf_ec2_run(run_args.split(), cfg)
    cluster_string = cluster_specs["cluster_string"]

    #time.sleep(run_time_sec)
    while not check_if_reached_iters(cluster_string, n_iters, cfg):
        time.sleep(60)

    tf_ec2_run(kill_args.split(), cfg)

    time.sleep(10)

    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, evaluator_file_name, outdir)
    tf_ec2_run(download_evaluator_file_args.split(), cfg)

    download_master_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, master_file_name, outdir)
    tf_ec2_run(download_master_file_args.split(), cfg)

def print_worker_sorted_times(fname):
    print("File: %s" % fname)
    print("-----------------------------")
    f = open(fname)

    compute_times = []
    for line in f:
        m = re.match(".*ELAPSED TIMES (.*)", line)
        if m:
            compute_times = eval(m.group(1))
    f.close()
    worker_times = {}
    for time, worker, iteration in compute_times:
        if worker not in worker_times:
            worker_times[worker] = []
        worker_times[worker].append(time)

    iteration_times = {}
    for time, worker, iteration in compute_times:
        if iteration not in iteration_times:
            iteration_times[iteration] = []
        iteration_times[iteration].append(time)

    percentile_100_times = []
    percentile_99_times = []
    percentile_95_times = []
    for iteration, times in iteration_times.items():
        percentile_99_times.append(np.percentile(times, 99, interpolation="nearest"))
        percentile_95_times.append(np.percentile(times, 95, interpolation="nearest"))
        percentile_100_times.append(np.percentile(times, 100, interpolation="nearest"))

    all_times = worker_times.values()
    a = []
    for time in all_times:
        a = a + time
    all_times = a

    print("Stdev: ", np.std(all_times))
    print("Max: ", np.max(all_times))
    print("80 Percentile:", np.percentile(all_times, 80))
    print("90 Percentile:", np.percentile(all_times, 90))
    print("95 Percentile:", np.percentile(all_times, 95))
    print("99 Percentile:", np.percentile(all_times, 99))
    print("Mean of 95 percentile (across iterations)", sum(percentile_95_times) / float(len(percentile_95_times)))
    print("Med of 95 percentile (across iterations)", np.median(percentile_95_times))
    print("Mean of 99 percentile (across iterations)", sum(percentile_99_times) / float(len(percentile_99_times)))
    print("Med of 99 percentile (across iterations)", np.median(percentile_99_times))
    print("Mean of 100 percentile (across iterations)", sum(percentile_100_times) / float(len(percentile_100_times)))
    print("Med of 100 percentile (across iterations)", np.median(percentile_100_times))
    print("Mean:", sum(all_times)/float(len(all_times)))

    print(percentile_99_times)

def extract_compute_times(fname):
    f = open(fname)

    compute_times = []
    for line in f:
        m = re.match(".*ELAPSED TIMES (.*)", line)
        if m:
            compute_times = eval(m.group(1))
    f.close()
    return [x[0] for x in compute_times]

def extract_compute_times_no_master(fname):
    f = open(fname)

    compute_times = []
    for line in f:
        m = re.match(".*ELAPSED TIMES (.*)", line)
        if m:
            compute_times = eval(m.group(1))
    f.close()
    return [x[0] for x in compute_times if x[1] != 14 and x[1] != 25]

def extract_iteration_times(fname):
    f = open(fname)

    compute_times = []
    for line in f:
        m = re.match(".*ITERATION TIMES (.*)", line)
        if m:
            compute_times = json.loads(m.group(1))
    f.close()
    return compute_times

def extract_times_losses_precision(fname):
    f = open(fname)

    times, losses, precisions, steps = [], [], [], []
    for line in f:
        m = re.match("Num examples: ([0-9]*)  Precision @ 1: ([\.0-9]*) Loss: ([\.0-9]*) Time: ([\.0-9]*)", line)
        step_match = re.match(".* step=([0-9]*)", line)
        if m:
            examples, precision, loss, time = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            times.append(time)
            losses.append(loss)
            precisions.append(precision)
        if step_match:
            step = int(step_match.group(1))
            steps.append(step)
    f.close()
    min_length = min([len(x) for x in [times, losses, precisions, steps]])
    return times[:min_length], losses[:min_length], precisions[:min_length], steps[:min_length]

def plot_time_precision(outdir):
    plt.cla()
    plt.xlabel("time (s)")
    plt.ylabel("precision (%)")
    files = glob.glob(outdir + "/*evaluator*")
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(files)))
    for i, fname in enumerate(files):
        label = fname.split("/")[-1]
        times, losses, precisions, steps = extract_times_losses_precision(fname)
        plt.plot(times, precisions, linestyle='solid', label=label, color=colors[i])
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig("time_precision.png")

def plot_step_loss(outdir):
    plt.cla()
    plt.xlabel("step")
    plt.ylabel("losses")
    files = glob.glob(outdir + "/*evaluator*")
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(files)))
    plt.yscale('log')
    #plt.xscale('log')
    for i, fname in enumerate(files):
        label = fname.split("/")[-1]
        times, losses, precisions, steps = extract_times_losses_precision(fname)
        plt.plot(steps, losses, linestyle='solid', label=label, color=colors[i])
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig("step_losses.png")

def plot_time_loss(outdir):
    plt.cla()
    plt.xlabel("time (s)")
    plt.ylabel("loss")
    files = glob.glob(outdir + "/*evaluator*")
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(files)))
    plt.yscale('log')
    #plt.xscale('log')
    for i, fname in enumerate(files):
        label = fname.split("/")[-1]
        times, losses, precisions, steps = extract_times_losses_precision(fname)
        plt.plot(times, losses, linestyle='solid', label=label, color=colors[i])
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig("time_loss.png")

def plot_time_step(outdir):
    plt.cla()
    plt.xlabel("time (s)")
    plt.ylabel("step")
    files = glob.glob(outdir + "/*evaluator*")
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 2.0, len(files)))
    for i, fname in enumerate(files):
        label = fname.split("/")[-1]
        times, losses, precisions, steps = extract_times_losses_precision(fname)
        #print(times, losses, precisions, steps)
        plt.plot(times, steps, linestyle='solid', label=label, color=colors[i])
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig("time_step.png")

def plot_time_cdfs(outdir):
    plt.cla()
    plt.xlabel("Time (s)")
    plt.ylabel("P(X <= x)")
    #plt.ylabel("Count")
    files = glob.glob(outdir + "/*t2*large*master*")
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(files) * 2))
    for i, fname in enumerate(files):
        label = fname.split("/")[-1]
        compute_times = extract_compute_times(fname)
        compute_times.sort()

        #plt.hist(compute_times, 50, alpha=.75)

        times = []
        probabs = []
        for compute_time in compute_times:
            times.append(compute_time)
            probabs.append(sum([1 if compute_time >= t else 0 for t in compute_times]) / float(len(compute_times)))
        plt.plot(times, probabs, linestyle='solid', label=label, color=colors[i])

        # Also plot the iteration times on top of the cdfs
        #times = []
        #probabs = []
        #iteration_times = extract_iteration_times(fname)
        #iteration_times = extract_compute_times_no_master(fname)
        #iteration_times.sort()
        #for iteration_time in iteration_times:
        #    times.append(iteration_time)
        #    probabs.append(sum([1 if iteration_time >= t else 0 for t in iteration_times]) / float(len(iteration_times)))
        #plt.plot(times, probabs, linestyle='solid', label=label + "_no_worker_25_14", color=colors[i + len(files)])

        print_worker_sorted_times(fname)

    plt.legend(loc="upper right", fontsize=6)
    #plt.savefig("histogram.png")
    plt.savefig("time_cdfs.png")

def plot_figs(cfgs, evaluator_file_name="out_evaluator", outdir="result_dir", n_iters=300, rerun=True, launch=True, need_shutdown_after_every_run=False):
    print([x["name"] for x in cfgs])
    if rerun:
        if launch and not need_shutdown_after_every_run:
            shutdown_and_launch(cfgs[0])
        for cfg in cfgs:
            if need_shutdown_after_every_run:
                shutdown_and_launch(cfg)
            run_tf_and_download_files(n_iters, cfg, evaluator_file_name=evaluator_file_name, outdir=outdir)

    plot_time_loss(outdir)
    plot_time_step(outdir)
    plot_time_precision(outdir)
    plot_step_loss(outdir)
    #plot_time_cdfs(outdir)

if __name__ == "__main__":
    print("Usage: python benchmark.py [use_dir dir|select_files cfg1 cfg2...] ")
    cfgs = []
    if len(sys.argv) >= 2:
        if sys.argv[1] == "use_dir":
            cfg_dir = sys.argv[2]
            cfg_filenames = glob.glob(cfg_dir + "/*")
            cfgs = [str(x) for x in cfg_filenames]
            cfgs = [load_cfg_from_file(x) for x in cfgs]
        elif sys.argv[1] == "select_files":
            cfgs = [load_cfg_from_file(x) for x in sys.argv[2:]]
    plot_figs(cfgs)
