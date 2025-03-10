import logging
from PIL import Image
from pathlib import Path
from rich.logging import RichHandler

class Logger:
    def __init__(self):
        self.name = "FlowDPS"

    def initLogger(self):
        __logger = logging.getLogger(self.name)

        FORMAT = f"[{self.name}] >> %(message)s"
        handler = RichHandler()
        handler.setFormatter(logging.Formatter(FORMAT))

        __logger.addHandler(handler)

        __logger.setLevel(logging.INFO)

        return __logger

def make_gif(input_path: Path, save_path: Path) -> None:
    files = sorted(input_path.glob('*.png'))
    frames = []

    for f in files:
        frames.append(Image.open(f).convert('RGB'))

    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
