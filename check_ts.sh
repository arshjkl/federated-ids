#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/frontend" && npx tsc --noEmit 2>&1 | grep -E "error TS" | head -20
