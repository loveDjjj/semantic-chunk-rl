"""Keep the original frozen VAE resident outside generation/training GPU processes."""

import argparse
import json
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
import torch

from scrl.runtime.transport import receive_message, send_message
from scrl.runtime.transport import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--socket-fd", type=int, required=True)
    args = p.parse_args()
    directory = Path(args.directory)
    sock = socket.socket(fileno=args.socket_fd)
    channel = sock.makefile("rwb")
    try:
        torch.cuda.set_device(args.device)
        torch.set_num_threads(2)
        settings = json.loads((directory / "config.json").read_text())
        sys.path.insert(0, settings["upstream"])
        from self_forcing import longlive_wan_wrapper as wrapper
        wrapper.WAN2_1_CHECKPOINT = settings["wan"]
        vae = wrapper.WanVAEWrapper().to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        ready = {"state": "ready", "device": args.device, "transport": "socket"}
        atomic_json(directory / "ready.json", ready)
        send_message(channel, ready)
        while True:
            message = receive_message(channel)
            if message["op"] == "stop":
                break
            request = message["request"]
            started = time.monotonic()
            latent = torch.load(request["latents"], map_location="cpu", weights_only=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    video = vae.decode_to_pixel(latent.to(args.device), use_cache=False)
                    pixels = ((video.float() * .5 + .5).clamp(0, 1) * 255).round().to(torch.uint8)
                    pixels = pixels[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()
                if len(pixels) != latent.shape[1] * 4 - 3:
                    raise RuntimeError("Wrong decoded frame extent")
                np.save(request["rgb"], pixels, allow_pickle=False)
                torch.cuda.synchronize()
                result = {"frames": len(pixels), "seconds": time.monotonic() - started,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                del latent, video, pixels
            finally:
                vae.model.clear_cache()
            send_message(channel, {"state": "complete", "result": result})
    except EOFError:
        pass
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
