import sys
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import matplotlib.pyplot as plt

def extract_time_losses(fname):
    f = open(fname)
    name = ""
    times, losses = [], []
    for i,line in enumerate(f):
        if i == 0:
            name = line
        else:
            step, time, loss, err = [float(x) for x in line.split(" ")]
            times.append(time)
            losses.append(loss)
    f.close()
    return name, times, losses

def plot_time_loss(fname):
    name, times, losses = extract_time_losses(fname)
    plt.plot(times, losses, label=name)

workdir = sys.argv[1]
for filename in glob.glob(workdir + "/time_loss_out*"):
    plot_time_loss(filename)
plt.legend(loc="upper right", fontsize=8)
plt.xlabel("Time (ms)", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.savefig("time_losses.png")
