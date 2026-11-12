"""JSON helpers and optional progress callbacks for the extracted runners."""
from pathlib import Path
import json
def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
class Progress:
    """Runner callback interface; override check/status for cancellation/reporting."""
    def check(self, force=False):
        pass
    def status(self, stage, **values):
        pass
