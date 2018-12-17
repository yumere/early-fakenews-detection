try:
    import ujson as json
except ImportError:
    import json

import argparse
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

from utils import bcolors


class D(object):
    def __init__(self, *args, **kwagrs):
        self.seq = np.array(args[0])
        self.text = args[1]
        self.rating = int(args[2])
        self.user_id = int(args[3])
        self.created_at = datetime.strptime(args[4], "%Y-%m-%d %H:%M:%S")

    @property
    def json(self):
        return (self.seq.tolist(), self.text, self.rating, self.user_id, str(self.created_at))

    def __str__(self):
        return "{seq} {rating} {text} {user_id} {created_at}".format(seq=str(self.seq),
                                                                     rating=self.rating,
                                                                     text=self.text,
                                                                     user_id=self.user_id,
                                                                     created_at=str(self.created_at))


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Input file", required=True)
    parser.add_argument('-d', '--pivot_date', type=valid_date, help="Pivot datetime", required=True)
    parser.add_argument('--ratio', type=float, help="Train ratio", default=0.7)
    parser.add_argument('--train_output', type=str, required=True)
    parser.add_argument('--dev_output', type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "rt", encoding="utf-8") as input_f:
        data = [D(*json.loads(i)) for i in input_f]

    print(bcolors.OKGREEN + "[+] Loaded {:,}".format(len(data)) + bcolors.ENDC)

    train_set = []
    dev_set = []

    for i, d in enumerate(tqdm(data, desc="Data", ncols=75)):
        if d.created_at < args.pivot_date:
            train_set.append(d)
        else:
            dev_set.append(d)

    print(bcolors.OKBLUE + "[*] Train set: {:,}".format(len(train_set)) + bcolors.ENDC)
    print(bcolors.OKBLUE + "[*] Dev set: {:,}".format(len(dev_set)) + bcolors.ENDC)

    train_user_set = set()
    dev_user_set = set()

    train_user_set.update([u.user_id for u in train_set])
    dev_user_set.update([u.user_id for u in dev_set])
    intersection_set = train_user_set.intersection(dev_user_set)
    print(bcolors.OKBLUE + "[*] Intersection users: {:,}".format(len(intersection_set)) + bcolors.ENDC)

    train_user_set -= intersection_set
    dev_user_set -= intersection_set

    sampled = set(random.sample(intersection_set, int(len(intersection_set)*args.ratio)))
    train_user_set.update(sampled)
    dev_user_set.update(intersection_set - sampled)

    print(bcolors.OKBLUE + "[*] Train user set: {:,}".format(len(train_user_set)) + bcolors.ENDC)
    print(bcolors.OKBLUE + "[*] Dev user set: {:,}".format(len(dev_user_set)) + bcolors.ENDC)

    train_set = list(filter(lambda x: x.user_id in train_user_set, train_set))
    dev_set = list(filter(lambda x: x.user_id in dev_user_set, dev_set))

    with open(args.train_output, "wt", encoding="utf-8") as f:
        for i, j in enumerate(tqdm(train_set, desc="Train set", ncols=75)):
            f.write(json.dumps(j.json) + "\n")

    tqdm.write(bcolors.OKGREEN + "Saved train set: {:,}".format(len(train_set)) + bcolors.ENDC)

    with open(args.dev_output, "wt", encoding="utf-8") as f:
        for i, j in enumerate(tqdm(dev_set, desc="Dev set", ncols=75)):
            f.write(json.dumps(j.json) + "\n")
    tqdm.write(bcolors.OKGREEN + "Saved dev set: {:,}".format(len(dev_set)) + bcolors.ENDC)
