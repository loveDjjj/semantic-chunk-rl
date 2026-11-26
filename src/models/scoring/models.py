"""Frozen reward model adapters, loaded in their own compatible environments."""

import importlib
import json
import math
import os
from pathlib import Path
import sys

import torch


def read_rgb(request):
    """Read one scope as uint8 THWC, without video encode/decode or duplicated files."""
    import numpy as np
    frames = np.load(request["rgb"], mmap_mode="r", allow_pickle=False)
    begin, end = request.get("start", 0), request.get("end", len(frames))
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
            or not 0 <= begin < end <= len(frames) or end - begin != request["frames"]):
        raise ValueError("Invalid RGB tensor or scope")
    return frames[begin:end]


class VBenchQuality:
    """Sampled quality proxy; not the official VBench aggregate."""

    def __init__(self, config, device):
        source, weights = Path(config["source"]), Path(config["weights"])
        sys.path.insert(0, str(source.resolve()))
        os.environ["VBENCH_CACHE_DIR"] = str(weights.resolve())
        import clip
        from pyiqa.archs.musiq_arch import MUSIQ
        from vbench.motion_smoothness import MotionSmoothness
        self.device = torch.device(device)
        self.clip, _ = clip.load(str(weights / "clip_model/ViT-L-14.pt"), device=self.device)
        aq = importlib.import_module("vbench.aesthetic_quality")
        self.imaging_module = importlib.import_module("vbench.imaging_quality")
        self.aesthetic = aq.get_aesthetic_model(str(weights / "aesthetic_model/emb_reader")).to(self.device)
        self.imaging = MUSIQ(pretrained_model_path=str(weights / "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth")).to(self.device)
        self.motion = MotionSmoothness(str(source / "vbench/third_party/amt/cfgs/AMT-S.yaml"),
                                      str(weights / "amt_model/amt-s.pth"), self.device)
        for model in (self.clip, self.aesthetic, self.imaging, self.motion.model):
            model.eval().requires_grad_(False)
        self.max_frames = int(config.get("max_frames", 32))
        self.batch_size = int(config.get('batch_size', 4))
        if not 1 <= self.batch_size <= 8:
            raise ValueError('Quality batch size must be in [1,8]')

    @torch.inference_mode()
    def score(self, request):
        if request.get('operation') == 'boundaries':
            return self.boundaries(request)
        from vbench.utils import clip_transform
        pixels = read_rgb(request) if "rgb" in request else None
        if pixels is None:
            import decord
            reader = decord.VideoReader(request["video"], num_threads=1)
        n = len(pixels) if pixels is not None else len(reader)
        if n < 3:
            raise ValueError("Insufficient unique frames for temporal quality")
        count = min(n, self.max_frames)
        indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
        selected = pixels[indices].copy() if pixels is not None else reader.get_batch(indices).asnumpy()
        frames = torch.from_numpy(selected).permute(0, 3, 1, 2)
        aesthetics, imaging, features = [], [], []
        for begin in range(0, count, self.batch_size):
            batch = frames[begin:begin + self.batch_size]
            encoded = self.clip.encode_image(clip_transform(224)(batch).to(self.device)).float()
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            features.append(encoded)
            aesthetics.append(self.aesthetic(encoded).flatten() / 10)
            transformed = self.imaging_module.transform(batch, "longer")
            imaging.append(self.imaging(transformed.to(self.device)).reshape(-1) / 100)
        features = torch.cat(features)
        # CLIP adjacency is an inexpensive consistency proxy, not official DINO subject consistency.
        # Compare at a fixed half-second separation, independent of video length.
        interval = max(1, round(request.get('fps', 16) * .5))
        interval = min(interval, n-1)
        starts = torch.linspace(0, n-1-interval, min(16, n-interval)).round().long().tolist()
        pairs = sorted(set(starts + [i+interval for i in starts]))
        paired = pixels[pairs].copy() if pixels is not None else reader.get_batch(pairs).asnumpy()
        paired = torch.from_numpy(paired).permute(0, 3, 1, 2)
        encoded = []
        for begin in range(0, len(pairs), self.batch_size):
            value = self.clip.encode_image(clip_transform(224)(paired[begin:begin+self.batch_size]).to(self.device)).float()
            encoded.append(torch.nn.functional.normalize(value, dim=-1))
        encoded = torch.cat(encoded)
        lookup = {frame: index for index, frame in enumerate(pairs)}
        consistency = (encoded[[lookup[i] for i in starts]] *
                       encoded[[lookup[i+interval] for i in starts]]).sum(-1).mean()
        if pixels is None:
            smoothness = float(self.motion.motion_score(request["video"]))
        else:
            # The worker serializes requests. Reuse upstream AMT arithmetic and all
            # original frames, replacing only this instance's file reader.
            reader_fn = self.motion.fp.get_frames
            self.motion.fp.get_frames = lambda _: list(pixels)
            try:
                smoothness = float(self.motion.motion_score("memory.mp4"))
            finally:
                self.motion.fp.get_frames = reader_fn
        scores = torch.stack([torch.cat(imaging).mean(), torch.cat(aesthetics).mean(), consistency]).cpu().tolist()
        raw = dict(zip(['imaging', 'aesthetic', 'consistency'], scores), motion=smoothness)
        if not all(math.isfinite(x) for x in raw.values()):
            raise RuntimeError("Nonfinite quality score")
        return {"quality": {k: min(1., max(0., v)) for k, v in raw.items()}, "raw": raw,
                "sampling": indices, "scope": "MUSIQ/AQ/CLIP consistency/AMT proxy"}

    @torch.inference_mode()
    def boundaries(self, request):
        """Fixed time windows; flow-compensated appearance change, not motion amount."""
        import cv2
        import numpy as np
        from vbench.utils import clip_transform
        pixels = read_rgb(request)
        width = max(2, round(request['fps']*.5))
        rows = []
        for boundary in request['boundaries']:
            if not 0 < boundary < len(pixels):
                raise ValueError('Boundary outside video')
            left = np.linspace(max(0, boundary-width), boundary-1, min(width, boundary)).round().astype(int)
            right = np.linspace(boundary, min(len(pixels)-1, boundary+width-1),
                                min(width, len(pixels)-boundary)).round().astype(int)
            frames = torch.from_numpy(pixels[np.concatenate([left, right])].copy()).permute(0, 3, 1, 2)
            features = []
            for begin in range(0, len(frames), self.batch_size):
                features.append(self.clip.encode_image(clip_transform(224)(
                    frames[begin:begin+self.batch_size]).to(self.device)).float())
            features = torch.nn.functional.normalize(torch.cat(features), dim=-1)
            sides = torch.stack([features[:len(left)].mean(0), features[len(left):].mean(0)])
            sides = torch.nn.functional.normalize(sides, dim=-1)
            appearance = float((sides[0]*sides[1]).sum().clamp(0, 1))
            def residual(index):
                first, second = [cv2.cvtColor(cv2.resize(pixels[i], (256,144)), cv2.COLOR_RGB2GRAY)
                                 for i in (index, index+1)]
                flow = cv2.calcOpticalFlowFarneback(first,second,None,.5,3,15,3,5,1.2,0)
                x,y=np.meshgrid(np.arange(256,dtype=np.float32),np.arange(144,dtype=np.float32))
                # Forward flow maps pixels in the first frame into the second.
                warped=cv2.remap(second,x+flow[:,:,0],y+flow[:,:,1],cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
                return float(np.abs(first.astype(np.float32)-warped).mean()/255.)
            crossing=residual(boundary-1)
            neighbors=sorted(set(max(0,min(len(pixels)-2,i)) for i in
                                 (boundary-width, boundary-2, boundary, boundary+width-2)))
            ordinary=float(np.median([residual(i) for i in neighbors]))
            motion=1./(1.+max(0.,crossing-ordinary*1.5)/.08)
            rows.append(dict(frame=boundary,appearance=appearance,motion=motion,
                             crossing_residual=crossing,neighbor_residual=ordinary))
        return {'boundary_quality': {key: sum(row[key] for row in rows)/len(rows)
                                     for key in ('appearance','motion')}, 'boundaries': rows}


class VideoAlignment:
    def __init__(self, config, device):
        source, weights = Path(config["source"]), Path(config["weights"])
        sys.path.insert(0, str(source.resolve()))
        import inference as official
        import train_reward
        from transformers import AutoConfig
        self.device = device
        model_config = json.loads((weights / "model_config.json").read_text())
        normalization = model_config.get("inference_config", {})
        self.center = float(config.get("center", normalization.get("TA_mean", 0)))
        self.scale = float(config.get("scale", normalization.get("TA_std", 1)))
        if self.scale <= 0:
            raise ValueError("Alignment scale must be positive")
        self.max_frames, self.max_pixels = int(config.get("max_frames", 32)), int(config.get("max_pixels", 131072))
        original_configs, original_load = official.load_configs_from_json, official.load_model_from_checkpoint
        cls = train_reward.Qwen2VLRewardModelBT
        original_descriptor = cls.__dict__.get("from_pretrained")

        def local_configs(path):
            result = list(original_configs(path))
            result[2] = dict(result[2], model_name_or_path=str(weights / "qwen2_vl_processor"))
            return tuple(result)

        def from_config(model_class, path, **kwargs):
            configuration = AutoConfig.from_pretrained(path, local_files_only=True)
            configuration.use_cache = False
            configuration._attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
            previous = torch.get_default_dtype()
            torch.set_default_dtype(kwargs.get("torch_dtype", torch.bfloat16))
            try:
                with torch.device("cpu"):
                    return model_class(configuration, output_dim=kwargs["output_dim"],
                        reward_token=kwargs["reward_token"], special_token_ids=kwargs["special_token_ids"])
            finally:
                torch.set_default_dtype(previous)

        def load(model, directory, step):
            state = torch.load(weights / "checkpoint-11352/model.pth", map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(state, strict=True)
            return model, "11352"

        official.load_configs_from_json, official.load_model_from_checkpoint = local_configs, load
        cls.from_pretrained = classmethod(from_config)
        try:
            self.scorer = official.VideoVLMRewardInference(str(weights), load_from_pretrained_step=11352,
                                                          device=device, dtype=torch.bfloat16)
        finally:
            official.load_configs_from_json, official.load_model_from_checkpoint = original_configs, original_load
            if original_descriptor is None:
                del cls.from_pretrained
            else:
                cls.from_pretrained = original_descriptor
        self.scorer.model.eval().requires_grad_(False)

    def prepare_rgb(self, request, count):
        """Match VideoAlign's uniform decoder sampling and uint8 bicubic resize exactly."""
        import vision_process as vision
        from prompt_template import build_prompt
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        if self.scorer.data_config.sample_type != "uniform":
            raise ValueError("RGB input currently requires VideoAlign uniform sampling")
        pixels = read_rgb(request)
        indices = torch.linspace(0, len(pixels) - 1, count).round().long().tolist()
        frames = torch.from_numpy(pixels[indices].copy()).permute(0, 3, 1, 2)
        height, width = frames.shape[-2:]
        size = vision.smart_resize(height, width, factor=vision.IMAGE_FACTOR,
                                   min_pixels=vision.VIDEO_MIN_PIXELS, max_pixels=self.max_pixels)
        frames = transforms.functional.resize(frames, list(size),
            interpolation=InterpolationMode.BICUBIC, antialias=True).float()
        data = [[{"role": "user", "content": [
            {"type": "video", "video": "memory.mp4", "max_pixels": self.max_pixels,
             "nframes": count, "sample_type": "uniform"},
            {"type": "text", "text": build_prompt(request["prompt"], self.scorer.data_config.eval_dim,
                                                  self.scorer.data_config.prompt_template_type)}]}]]
        processor = self.scorer.processor
        batch = processor(text=processor.apply_chat_template(data, tokenize=False, add_generation_prompt=True),
                          images=None, videos=[frames], padding=True, return_tensors="pt",
                          videos_kwargs={"do_rescale": True})
        return self.scorer._prepare_inputs(batch)

    @torch.inference_mode()
    def score(self, request):
        count = min(request["frames"], self.max_frames)
        count -= count % 2
        if count < 2:
            raise ValueError("Not enough distinct alignment frames")
        batch = (self.prepare_rgb(request, count) if "rgb" in request else
                 self.scorer.prepare_batch([request["video"]], [request["prompt"]],
                                           num_frames=count, max_pixels=self.max_pixels))
        if int(batch["attention_mask"][0].sum()) > 6144:
            raise ValueError("VideoReward context exceeds its trained length")
        ids = batch["input_ids"][0]
        special = self.scorer.processor.tokenizer.convert_tokens_to_ids(
            ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"])
        if any(int((ids == token).sum()) != 1 for token in special):
            raise RuntimeError("Missing VideoReward special tokens")
        raw = self.scorer.model(**batch, return_dict=True)["logits"][0].float()
        if raw.shape != (3,) or not torch.isfinite(raw).all():
            raise RuntimeError("Invalid alignment output")
        ta = float(raw[2])
        alignment = 1 / (1 + math.exp(-max(-40, min(40, (ta - self.center) / self.scale))))
        return {"semantic": {"alignment": alignment}, "raw": dict(zip(["VQ", "MQ", "TA"], raw.tolist())),
                "sampled_frames": count, "scope": "VideoReward text-video alignment proxy"}
