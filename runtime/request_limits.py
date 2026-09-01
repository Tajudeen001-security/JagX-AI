from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RequestLimits:
    max_input_tokens:int=16384
    max_output_tokens:int=4096
    max_batch_size:int=8
    max_image_bytes:int=25*1024*1024
    max_video_seconds:int=120
    max_concurrent_jobs:int=16

    def validate_text(self,input_tokens:int,output_tokens:int):
        if not 0 <= input_tokens <= self.max_input_tokens: raise ValueError('input token limit exceeded')
        if not 0 <= output_tokens <= self.max_output_tokens: raise ValueError('output token limit exceeded')

    def validate_video(self,duration_seconds:int):
        if not 0 <= duration_seconds <= self.max_video_seconds: raise ValueError('video request limit exceeded')
