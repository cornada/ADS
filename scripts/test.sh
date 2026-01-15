#!/usr/bin/env bash
# Run pytest with PYTHONPATH set to packages directory
# Usage: ./scripts/test.sh [pytest args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_ROOT/packages:$PYTHONPATH"

cd "$PROJECT_ROOT"
pytest -q "$@"
