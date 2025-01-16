"""
This type stub file was generated by pyright.
"""

class SoundDeviceError(Exception):
    ...


class Voice:
    max_rms = ...
    min_rms = ...
    pct = ...
    threshold = ...
    def __init__(self, audio_format=..., device_name=...) -> None:
        ...
    
    def callback(self, indata, frames, time, status): # -> None:
        """This is called (from a separate thread) for each audio block."""
        ...
    
    def get_prompt(self): # -> str:
        ...
    
    def record_and_transcribe(self, history=..., language=...): # -> Any | None:
        ...
    
    def raw_record_and_transcribe(self, history, language): # -> Any | None:
        ...
    


if __name__ == "__main__":
    api_key = ...
