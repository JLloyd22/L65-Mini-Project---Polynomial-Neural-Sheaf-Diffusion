#! /bin/zsh
#source this file to setup the environment variables for macOS
export LDFLAGS="${LDFLAGS/--no-as-needed/}"
uv sync
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PYTHONPATH"