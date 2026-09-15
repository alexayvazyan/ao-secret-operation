#!/usr/bin/env bash
# Render each figure HTML to a 1600px-wide PNG with Windows Chrome: measure page height from the DOM, then screenshot.
set -euo pipefail
W=/mnt/c/Users/alexa/AppData/Local/Temp/aofig
CHROME="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
cd "$W"
for f in "$@"; do
  url="file:///$(wslpath -w "$W/$f" | sed 's#\\#/#g')"
  h=$(timeout 60 "$CHROME" --headless=new --disable-gpu --virtual-time-budget=3000 --window-size=1600,1000 --dump-dom "$url" 2>/dev/null \
      | grep -o 'data-height="[0-9]*"' | grep -o '[0-9]*' | head -1)
  timeout 60 "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --virtual-time-budget=3000 \
    --window-size=1600,"$h" --screenshot="$(wslpath -w "$W/${f%.html}.png")" "$url" >/dev/null 2>&1
  echo "$f -> ${f%.html}.png (1600x$h)"
done
