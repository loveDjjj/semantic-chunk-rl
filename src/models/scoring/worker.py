"""Resident GPU scorer with blocking socket control and mmap RGB input."""

import argparse
import json
from pathlib import Path
import socket
import time
import traceback
import torch

from scrl.runtime.transport import receive_message, send_message
from scrl.models.scoring.models import VBenchQuality, VideoAlignment
from scrl.runtime.transport import atomic_json
from scrl.rl.reward import UnscorableReward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["quality", "alignment", "judge"], required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--socket-fd", type=int, required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    sock = socket.socket(fileno=args.socket_fd)
    channel = sock.makefile("rwb")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2)
    settings = json.loads((directory / "config.json").read_text())
    try:
        if args.kind == 'judge':
            from scrl.models.scoring.video_judge import VideoJudge
            model = VideoJudge(settings, args.device)
        else:
            model = (VBenchQuality if args.kind == "quality" else VideoAlignment)(settings, args.device)
        ready = {"state": "ready", "device": args.device, "transport": "socket"}
        atomic_json(directory / "ready.json", ready)
        send_message(channel, ready)
        while True:
            try:
                message = receive_message(channel)
            except EOFError:
                break
            if message["op"] == "stop":
                break
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            try:
                result = model.score(message["request"])
            except UnscorableReward as error:
                torch.cuda.synchronize()
                send_message(channel, {"state": "unscorable", "error": str(error)})
                continue
            torch.cuda.synchronize()
            result["seconds"] = time.monotonic() - started
            result['peak_memory_mib'] = torch.cuda.max_memory_allocated() / 2**20
            send_message(channel, {"state": "complete", "result": result})
    except BaseException as error:
        response = {"state": "failed", "error": repr(error), "traceback": traceback.format_exc()}
        atomic_json(directory / "failure.json", response)
        try:
            send_message(channel, response)
        except OSError:
            pass
        raise
    finally:
        channel.close()
        sock.close()


if __name__ == "__main__":
    main()
