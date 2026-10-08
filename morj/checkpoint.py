"""Find the checkpoint to serve, downloading the pinned revision when there is no local copy.

Two checkpoints of the same model are supported, chosen with MORJ_CHECKPOINT:

- `bf16`: google/gemma-4-31B-it, BF16 (any GPU with 80 GB+).
- `nvfp4`: nvidia/Gemma-4-31B-IT-NVFP4, NVIDIA's Model Optimizer NVFP4 quantization (Blackwell).

Order: MORJ_MODEL_PATH if set; else a checkpoint mounted at /model (docker compose mounts
MODEL_DIR there); else the pinned revision from the Hugging Face cache, downloaded on first use.
Startup then checks the checkpoint's fingerprints against provenance/identity-<checkpoint>.json,
so other revisions are refused.

Prompts are built with Google's canonical tokenizer_config.json and chat_template.jinja from
google/gemma-4-31B-it (revision 842da37), shipped in provenance/tokenizer/ and combined with the
checkpoint's tokenizer.json (byte-identical in both repositories) into a tokenizer overlay. The
NVFP4 repository carries an older chat template; for this service's prompts both render the same
tokens, and the overlay keeps the two checkpoints on one set of files.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROVENANCE = Path(__file__).with_name('provenance')
TOKENIZER_FILES = PROVENANCE / 'tokenizer'
FILES = ['*.json', '*.jinja', '*.safetensors']
MOUNT = Path('/model')


@dataclass(frozen=True)
class Checkpoint:
    name: str
    repo: str
    revision: str
    size: str
    precision: str
    model_name: str      # served model name, as in the release this checkpoint was evaluated with
    needs_fp4: bool      # NVFP4 GEMMs need FP4 tensor cores (Blackwell) unless Marlin is selected

    @property
    def identity_file(self):
        return PROVENANCE / f'identity-{self.name}.json'

    def identity(self):
        return json.loads(self.identity_file.read_text())


CHECKPOINTS = {
    'bf16': Checkpoint('bf16', 'google/gemma-4-31B-it', '842da3794eaa0b77d5f08bae87a17459d91ff475', 'about 62 GB',
                       'BF16', 'gemma-4-31b-trt', False),
    'nvfp4': Checkpoint('nvfp4', 'nvidia/Gemma-4-31B-IT-NVFP4', '4135a98a9b728a548947683219633b25682223ac',
                        'about 33 GB', 'NVFP4 MLP (weights and activations), BF16 attention, FP8 KV cache',
                        'gemma-4-31b-nvfp4-trt', True),
}


def say(message):
    print(f'morj: {message}', file=sys.stderr, flush=True)


def access_error(checkpoint):
    return (f'Hugging Face refused access to {checkpoint.repo}. Check that https://huggingface.co/{checkpoint.repo} '
            'opens for your account, then export HF_TOKEN=hf_... (or run `hf auth login` on this machine) and start again.')


def complete(path):
    """A checkpoint directory with its config and every weight shard the index names."""
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if not (path / 'config.json').is_file() or not index.is_file():
        return False
    shards = set(json.loads(index.read_text())['weight_map'].values())
    return all((path / shard).is_file() for shard in shards)


def resolve(checkpoint, model_path=''):
    """Directory of the checkpoint to serve."""
    if model_path:
        if not complete(model_path):
            raise SystemExit(f'MORJ_MODEL_PATH={model_path} is not a complete checkpoint directory')
        return model_path
    if complete(MOUNT):
        return str(MOUNT)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    repo, revision = checkpoint.repo, checkpoint.revision
    try:
        cached = snapshot_download(repo, revision=revision, allow_patterns=FILES, local_files_only=True)
        if complete(cached):
            return cached
    except Exception:  # nothing cached yet
        pass
    say(f'downloading {repo} at revision {revision[:7]} into the Hugging Face cache '
        f'({checkpoint.size}; an interrupted download resumes where it stopped)')
    try:
        path = snapshot_download(repo, revision=revision, allow_patterns=FILES)
    except GatedRepoError:
        raise SystemExit(access_error(checkpoint)) from None
    except HfHubHTTPError as error:
        if getattr(error.response, 'status_code', None) in (401, 403):
            raise SystemExit(access_error(checkpoint)) from None
        raise
    if not complete(path):
        raise SystemExit(f'Download of {repo} is incomplete; start again to resume it')
    return path


def tokenizer_overlay(model_path, directory, expected):
    """A tokenizer directory: the checkpoint's tokenizer.json plus the canonical config and chat template.

    `expected` maps each of the three file names to its SHA-256; a mismatch refuses to start.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = {'tokenizer.json': Path(model_path) / 'tokenizer.json',
               'tokenizer_config.json': TOKENIZER_FILES / 'tokenizer_config.json',
               'chat_template.jinja': TOKENIZER_FILES / 'chat_template.jinja'}
    for name, source in sources.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f'{name} ({source}) differs from the evaluated tokenizer files')
        shutil.copyfile(source, directory / name)
    return str(directory)
