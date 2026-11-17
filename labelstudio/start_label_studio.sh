#!/bin/bash
# Start Label Studio with everything on D.
# Usage (Git Bash):  bash /d/Thesis/start_label_studio.sh practice   (or pilot_r1, full/full_geron_dev, full/full_geron_test)
# Stop: Ctrl+C in this window.
PKG="${1:-practice}"
PKG="$(echo "$PKG" | tr '\\' '/')"          # accept full\full_geron_dev or full/full_geron_dev
NAME="$(basename "$PKG")"
ROOT="D:\\Thesis\\baguio_annotation\\$(echo "$PKG" | tr '/' '\\')"
if [ ! -d "/d/Thesis/baguio_annotation/$PKG/images" ]; then
  echo "No images folder at $ROOT\\images - check the package name."; exit 1
fi
export TMP='D:\tmp' TEMP='D:\tmp' PIP_CACHE_DIR='D:\pip-cache'
export LABEL_STUDIO_BASE_DATA_DIR='D:\label-studio-data'
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# Only people with an invite link can create an account (the server is reachable on the local network).
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT"
echo "Package:        $PKG"
echo "Storage path:   $ROOT\\images   (use this in Settings > Cloud Storage > Local files)"
echo "Tasks to import: $ROOT\\labelstudio_tasks_$NAME.json"
echo "Opening http://localhost:8080 ..."
/d/ls-env/Scripts/label-studio.exe start --port 8080
