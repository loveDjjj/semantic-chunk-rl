"""Lightweight throughput diagnostics, restricted to assigned physical GPU IDs."""
import csv
from pathlib import Path
import subprocess
import time


class SpeedSampler:
    def __init__(self, devices):
        self.devices = list(devices)
        self.last = None

    def sample(self, node):
        row = dict(unix=time.time(), node=node, gpus=[])
        fields = 'index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total'
        try:
            data = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str,self.devices)),
                '--query-gpu='+fields, '--format=csv,noheader,nounits'], text=True,
                timeout=3, stderr=subprocess.DEVNULL)
            for i, uuid, *values in csv.reader(data.splitlines(), skipinitialspace=True):
                if int(i) not in self.devices:
                    continue
                vals = [float(x) if x.strip().replace('.', '', 1).isdigit() else None for x in values]
                util, busy, used, total = vals
                row['gpus'].append(dict(index=int(i), uuid=uuid, utilization=util,
                    memory_busy_pct=busy, memory_mib=used, memory_total_mib=total,
                    memory_occupied_pct=100*used/total if total and used is not None else None))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            row['gpu_sample_error'] = type(exc).__name__
        try:
            cpu = list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
            net = {}
            for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
                name, data = line.split(':')
                if name.strip() != 'lo':
                    counters = data.split()
                    net[name.strip()] = (int(counters[0]), int(counters[8]))
            now = time.monotonic()
            if self.last:
                t, old, interfaces = self.last
                delta = [a-b for a,b in zip(cpu,old)]
                total = sum(delta)
                if total>0 and min(delta)>=0:
                    row.update(cpu_busy_pct=100*(total-delta[3]-delta[4])/total,
                               cpu_iowait_pct=100*delta[4]/total)
                row['network_bytes_per_sec'] = {
                    name: dict(rx=(rx-interfaces[name][0])/(now-t),tx=(tx-interfaces[name][1])/(now-t))
                    for name,(rx,tx) in net.items() if now>t and name in interfaces
                    and rx>=interfaces[name][0] and tx>=interfaces[name][1]}
            self.last = now,cpu,net
        except (OSError,ValueError,IndexError) as exc:
            row['host_sample_error'] = type(exc).__name__
        return row
