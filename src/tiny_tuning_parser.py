import re
import argparse

parser = argparse.ArgumentParser(description='Distributed Tunning')
parser.add_argument('--tuning-dir', type=str, default='./tune/', metavar='N',
                        help='directory to save temp tuning logs')
parser.add_argument('--tuning-lr', type=float, default=0.125, metavar='N',
                        help='candidate learning rate used during tunning')
parser.add_argument('--num-workers', type=int, default=16, 
                        help='number of workers in the cluster')
args = parser.parse_args()

loss_stat = []
with open(args.tuning_dir, 'rb') as file:
    for line in file.readlines():
        line_content = line.rstrip('\n')
        search = re.search(
            'Worker: .*, Step: .*, Epoch: .* \[.* \(.*\)\], Loss: (.*), Time Cost: .*, Comp: .*, Encode:  .*, Comm:  .*, Msg\(MB\):  .*', 
            line_content)
        if search:
            loss = float(search.group(1))
            loss_stat.append(loss)
    try:
        assert len(loss_stat) == args.num_workers
    except AssertionError:
        print("Illeagel Number of Workers! ")
    print("Avged loss for lr candidate: {}=========>{}".format(args.tuning_lr, sum(loss_stat)/float(len(loss_stat))))