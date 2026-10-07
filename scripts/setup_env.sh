#!/usr/bin/env bash
# Create .venv (Python 3.12 via uv) and make LightGBM load scikit-learn's bundled
# libomp, since this Mac has no Homebrew libomp.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
if [[ "$(uname)" == "Darwin" ]]; then
  SP=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  LIB="$SP/lightgbm/lib/lib_lightgbm.dylib"
  install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs "$LIB" 2>/dev/null || true
  codesign -f -s - "$LIB"
fi
.venv/bin/python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'ok')"
