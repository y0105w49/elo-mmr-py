#!/usr/bin/env python3

from collections import defaultdict
import glob
import json
from pprint import pprint as pp

from elo_mmr import rate


dir = '../Elo-MMR/cache/dmoj'
fs = glob.glob(dir+'/*.json')
fs = sorted(fs, key=lambda f:int(f.split('/')[-1][:-5]))

hists = defaultdict(list)

for f in fs:
    with open(f) as fp:
        o = json.load(fp)
        name = o['name']
        ts = o['time_seconds']
        stands = o['standings']
        weight = o['weight'] if 'weight' in o else 1.

        print(f'rating {name} w/ {len(stands)} users')
        uhistss = {u: hists[u] for u, rlo, rhi in stands}
        rate(name, ts, stands, uhistss, weight=weight)
        # for u, nhists in nhists.items():
            # hists[u].append(nhist[-1])

        # if name == 'gfssoc1j':
            # break

# pp(hists)
# exit(0)

lb = []
for u,h in hists.items():
    lb.append((h[-1].rating, u, h[-1]))
lb = sorted(lb, reverse=True)
pp(lb[::-1])

pp(hists[lb[0][1]])
