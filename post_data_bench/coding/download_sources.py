"""Download missing LCB shards at one resolved HF commit, publishing atomically."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from gold_generation import DATA_DIR, SOURCE_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    missing = [name for name in SOURCE_FILES if not (DATA_DIR/name).is_file()]
    if not missing:
        print('All frozen source shards are present; no download.')
        return
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'livecodebench/code_generation_lite'
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    print(f'Downloading missing shards from {repo}@{revision}')
    for name in missing:
        cached = hf_hub_download(repo_id=repo, repo_type='dataset', revision=revision, filename=name)
        # Validate cached downloads before publishing their final names.
        with open(cached) as stream:
            count = sum(bool(json.loads(line)) for line in stream if line.strip())
        if not count:
            raise ValueError(f'Empty source shard: {name}')
        with tempfile.NamedTemporaryFile(dir=DATA_DIR, prefix=name+'.', suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(cached, temporary)
            os.replace(temporary, DATA_DIR/name)
        finally:
            temporary.unlink(missing_ok=True)
        (DATA_DIR/(name+'.revision')).write_text(revision+'\n')


if __name__ == '__main__':
    main()
