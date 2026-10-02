import runpy, sys
from pathlib import Path
SRC = "/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/wt/src"
sys.path.insert(0, SRC); sys.path.insert(0, SRC + "/signal-and-noise")
import pretrain.launch_trainings as LT
LT.DATA_MASTER = Path(sys.argv[1])
script = sys.argv[2]; sys.argv = [script, *sys.argv[3:]]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
