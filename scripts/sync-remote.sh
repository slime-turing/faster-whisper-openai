#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <user@host> [remote_dir]" >&2
  exit 1
fi

remote_host="$1"
remote_dir="${2:-~/compose/faster-whisper}"

rsync -av --delete \
  --exclude '.git/' \
  --exclude '.github/' \
  --exclude '.env' \
  --exclude 'cache/' \
  --exclude 'tmp/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.venv/' \
  ./ "$remote_host:$remote_dir/"