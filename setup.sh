#!/bin/bash
# Fetch the third-party data used by the model, at the exact commits used for the published results:
#   RTS-GMLC       https://github.com/GridMod/RTS-GMLC            commit 3ece0d3
#   PDSA-RTS-GMLC  https://github.com/FredericSabot/PDSA-RTS-GMLC commit 9edac7e
# The code expects them as ./RTS-GMLC and ./PDSA-RTS-GMLC in the repository root.
#
# Usage, from the repository root:
#   bash setup.sh            clone both repositories here (about 560 MB)
#   MOF1_DATA=DIR bash setup.sh
#                            clone them into DIR and create symbolic links here
#   bash setup.sh --venv     also create a virtual environment .venv with the pinned packages of
#                            requirements.txt (needs python3.12 with the venv module)
set -euo pipefail
cd "$(dirname "$0")"
DATA=${MOF1_DATA:-.}
mkdir -p "$DATA"

fetch() {
  local url=$1 dir=$2 commit=$3 dest="$DATA/$2"
  if [ ! -d "$dest/.git" ]; then
    echo "cloning $url into $dest"
    git clone --quiet "$url" "$dest"
  fi
  if ! git -C "$dest" cat-file -e "$commit^{commit}" 2> /dev/null; then
    git -C "$dest" fetch --quiet origin
  fi
  git -C "$dest" -c advice.detachedHead=false checkout --quiet "$commit"
  local head; head=$(git -C "$dest" rev-parse --short=7 HEAD)
  [ "$head" = "$commit" ] || { echo "error: $dest is at $head, expected $commit"; exit 1; }
  if [ "$DATA" != "." ] && [ ! -e "$dir" ]; then
    ln -s "$(cd "$DATA" && pwd)/$dir" "$dir"
  fi
  echo "$dir at commit $head"
}

fetch https://github.com/GridMod/RTS-GMLC RTS-GMLC 3ece0d3
fetch https://github.com/FredericSabot/PDSA-RTS-GMLC PDSA-RTS-GMLC 9edac7e

[ -d RTS-GMLC/RTS_Data/SourceData ] || { echo "error: RTS-GMLC/RTS_Data/SourceData not found"; exit 1; }
[ -f PDSA-RTS-GMLC/3-DynData/RTS.par ] || { echo "error: PDSA-RTS-GMLC/3-DynData/RTS.par not found"; exit 1; }

if [ "${1:-}" = "--venv" ]; then
  PYBIN=${PYTHON:-python3.12}
  "$PYBIN" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  .venv/bin/python -c 'import sys, scipy, numpy, pandas, matplotlib; print("python %s, scipy %s, numpy %s, pandas %s, matplotlib %s" % (sys.version.split()[0], scipy.__version__, numpy.__version__, pandas.__version__, matplotlib.__version__))'
  echo "use it with:  export PY=$(pwd)/.venv/bin/python"
fi
echo "setup complete"
