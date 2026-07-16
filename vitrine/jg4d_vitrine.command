#!/bin/bash
# Double-click this to start the jg4d vitrine app.
# macOS opens .command files in Terminal, which is what we want: the log lives
# there, and closing the window quits the app.

cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 /usr/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    echo "No python3 found. Install it from https://www.python.org/downloads/ and try again."
    read -r -p "Press return to close."
    exit 1
fi

if ! "$PY" -c "import numpy, PIL" 2>/dev/null; then
    echo "Missing numpy/Pillow. Installing them now..."
    "$PY" -m pip install --user numpy pillow || {
        echo "Install failed. Run this by hand:  $PY -m pip install numpy pillow"
        read -r -p "Press return to close."
        exit 1
    }
fi

echo "Starting jg4d vitrine ..."
"$PY" jg4d_app.py
echo
read -r -p "App stopped. Press return to close this window."
