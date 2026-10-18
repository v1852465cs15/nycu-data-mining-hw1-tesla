import mimetypes

import trackio

from utils import read_settings

mimetypes.add_type("text/javascript", ".js")

trackio.show(project=read_settings()["trackio"]["project"], block_thread=True)
