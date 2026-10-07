"""Build Ray's runtime environment without printing judge credentials."""
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    runtime = json.loads(sys.argv[1])
    if os.environ.get("RUBRIC_REWARDS") == "1":
        for key in ("LITELLM_JUDGE_API_KEY", "LITELLM_JUDGE_BASE_URL", "LITELLM_BASE_URL", "JUDGE_MAX_CONSECUTIVE_FAILURES", "JUDGE_QUOTA_STOP_FILE"):
            if key in os.environ:
                runtime["env_vars"][key] = os.environ[key]
    target = Path(sys.argv[2])
    # Reclaim same-user files from dead launcher processes without reading secrets.
    # /dev/shm is volatile; SIGKILL cleanup happens on the next launch or reboot.
    for orphan in target.parent.glob('rubric-ray-env.*.*.json'):
        fields = orphan.name.split('.')
        if orphan == target or not fields[1].isdigit() or orphan.lstat().st_uid != os.getuid():
            continue
        try:
            os.kill(int(fields[1]), 0)
        except ProcessLookupError:
            orphan.unlink(missing_ok=True)
        except PermissionError:
            pass
    target.chmod(0o600)
    target.write_text(json.dumps(runtime))
