import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

fname = sys.argv[1]
f = open(fname)

master_times = {}
worker_times = {}
steps = set()
name = ""

for i, line in enumerate(f):
    if i == 0:
        name = line
    else:
        time, step, is_master = (int(x) for x in line.split(" "))
        steps.add(step)
        if is_master:
            assert(step not in master_times)
            master_times[step] = time
        else:
            if step not in worker_times:
                worker_times[step] = []
            worker_times[step].append(time)

base_index = 0
linewidth = 3
increments = linewidth + 20
plt.gca().yaxis.set_major_locator(plt.NullLocator())

for step in sorted(list(steps)):
    if step not in worker_times or step not in master_times:
        break

    if step-1 not in worker_times:
        begin_times = [0] * len(worker_times[step])
    else:
        begin_times = [x+50 for x in sorted(worker_times[step-1])]
    worker_times_for_step = sorted(worker_times[step])
    master_time_for_step = master_times[step]
    plt.hlines([base_index + x * increments for x in range(len(worker_times_for_step))],
               begin_times,
               worker_times_for_step,
               linewidth=linewidth)
    plt.axvline(x=master_time_for_step)

avg_gradients_received = 0
print(name)
for i, step in enumerate(sorted(list(steps))):
    if i == len(list(steps))-1:
        break
    if step not in worker_times or step not in master_times:
        break
    count = len(worker_times[step])
    avg_gradients_received += count
    print("Step: %d, NGradientsReceived: %d" % (step, count))
avg_gradients_received /= float(len(list(steps))-1)

print("Average number of gradients received per step: %f" % avg_gradients_received)

plt.ylabel("Sorted worker iteration times")
plt.xlabel("Time (ms)")
plt.savefig(name + ".png")

f.close()
