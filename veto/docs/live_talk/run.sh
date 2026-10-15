#!/bin/sh
# Build the live veto talk and present it with PictureThis (macOS). The app starts veto_talk.py itself.
#   ./run.sh                 build + present
#   ./run.sh --page 12       start on a slide
set -e
here="$(cd "$(dirname "$0")" && pwd)"
pt="${PICTURE_THIS:-$HOME/workspace/uzh/picture-this}"
(cd "$here" && TEXINPUTS="$here//:$pt/latex//:" latexmk -lualatex -interaction=nonstopmode -silent veto_talk.tex)
swift build --package-path "$pt" -c release >/dev/null
PYTHONPATH="$pt/python${PYTHONPATH:+:$PYTHONPATH}" exec "$pt/.build/release/PictureThis" "$here/veto_talk.pdf" \
  --backend "$here/veto_talk.py" --python "$(command -v python3)" "$@"
