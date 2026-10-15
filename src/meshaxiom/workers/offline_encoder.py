"""Construct the DINOv3 architecture locally, then load its released EMA tensors."""
import contextlib
from unittest.mock import patch


@contextlib.contextmanager
def offline_encoder(directory):
    from transformers import AutoConfig, AutoModel

    def build(*unused, **ignored):
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        return AutoModel.from_config(config)

    # Stage workers are isolated and serial. No globally installed library is edited.
    with patch.object(AutoModel, "from_pretrained", new=build):
        yield
