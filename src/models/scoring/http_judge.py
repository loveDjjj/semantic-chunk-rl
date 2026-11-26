"""Node-local vLLM transport for the existing global/segment reward protocol."""

from concurrent.futures import ThreadPoolExecutor
import base64
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


from scrl.rl.reward import UnscorableReward
from scrl.models.scoring.grounding import event_batch_size
from scrl.models.scoring.protocol import parse_object
from scrl.models.scoring.video_judge import VideoJudge


class UserText:
    def apply_chat_template(self, messages, **kwargs):
        # The scoring policy may insert a repair instruction before the terminator.
        return next(p['text'] for p in messages[0]['content'] if p['type']=='text') + '<|im_end|>'


def png(frame):
    import cv2
    ok, data = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError('Failed to encode judge RGB frame')
    return base64.b64encode(data).decode('ascii')


class HttpJudge(VideoJudge):
    def __init__(self, config, port, limit):
        self.base = f'http://127.0.0.1:{int(port)}'
        self.processor = UserText()
        self.template_options = {}
        self.max_frames = config['max_frames']
        self.event_choice_batch_size = event_batch_size(config)
        self.max_tokens = config['max_new_tokens']
        self.pixel_budget = config['pixels_per_frame']
        self.batch_size = 2
        self.cache = Path(config['contract_cache'])
        from scrl.models.scoring.cache_client import cache_peer
        self.event_cache_peer = cache_peer(config)
        self.event_decisions = config.get('event_decisions')
        self.cache.mkdir(exist_ok=True, parents=True)
        self.slots = threading.BoundedSemaphore(limit)

    def ready(self):
        with urllib.request.urlopen(self.base+'/health', timeout=10):
            return True

    def close(self):
        pass

    def request_text(self, text, video=None, metadata=None, image=None, limit=None, constraint=None):
        content = []
        req = dict(model='rl-video-judge', temperature=0, max_tokens=limit or self.max_tokens,
                   chat_template_kwargs={'enable_thinking':False})
        if constraint:
            req['structured_outputs'] = constraint
        if video is not None:
            # vLLM's frame-sequence loader decodes each element with PIL, including lossless PNG.
            url = 'data:video/jpeg;base64,' + ','.join(map(png, video))
            content.append(dict(type='video_url',video_url={'url':url}))
            req['media_io_kwargs'] = {'video': dict(num_frames=len(video), fps=metadata.fps,
                frames_indices=metadata.frames_indices,total_num_frames=metadata.total_num_frames,
                do_sample_frames=False)}
            req['mm_processor_kwargs'] = dict(do_sample_frames=False, size={
                'shortest_edge':128*32*32, 'longest_edge':self.max_frames*self.pixel_budget})
        elif image is not None:
            content.append(dict(type='image_url',image_url={'url':'data:image/png;base64,'+png(image)}))
        content.append(dict(type='text',text=text.removesuffix('<|im_end|>')))
        req['messages'] = [dict(role='user',content=content)]
        data = json.dumps(req).encode()
        request = urllib.request.Request(self.base+'/v1/chat/completions', data=data,
                                          headers={'Content-Type':'application/json'})
        try:
            with self.slots, urllib.request.urlopen(request, timeout=300) as reply:
                return json.load(reply)['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'vLLM HTTP {e.code}: {e.read(1000)!r}') from e

    def _generate(self, texts, videos=None, metadata=None, *, repair=True, return_invalid=False):
        values = []
        for i,text in enumerate(texts):
            raw = self.request_text(text, videos[i] if videos else None, metadata[i] if metadata else None)
            try:
                values.append(parse_object(raw))
            except UnscorableReward as error:
                if not return_invalid:
                    raise
                values.append({'invalid_json':raw, 'parse_error':str(error)})
        return values

    def score(self, request):
        started = time.monotonic()
        result = super().score(request)
        result['seconds'] = time.monotonic()-started
        return result


def parallel_class():
    # Only the scoring runtime ships OpenCV.

    class ParallelScopes(HttpJudge):
        def __init__(self, settings, port, limit):
            super().__init__(settings,port,limit)
            self.scope_pool=ThreadPoolExecutor(max_workers=limit,thread_name_prefix='judge-scope')

        def _generate(self, texts, videos=None, metadata=None, **kwargs):
            generate=super()._generate
            if len(texts)!=2 or videos is None or metadata is None:
                return generate(texts,videos,metadata,**kwargs)
            second=self.scope_pool.submit(generate,[texts[1]],[videos[1]],[metadata[1]],**kwargs)
            try:
                first=generate([texts[0]],[videos[0]],[metadata[0]],**kwargs)
            except BaseException:
                # A parallel scope's invalid evidence must not mask a transport/model failure.
                try:
                    second.result()
                except UnscorableReward:
                    pass
                raise
            return first+second.result()

        def close(self):
            self.scope_pool.shutdown(wait=True)
            super().close()

    return ParallelScopes
