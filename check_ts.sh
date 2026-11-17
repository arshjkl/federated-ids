#!/bin/bash
cd /Users/admin/Desktop/federated-ids/frontend && npx tsc --noEmit 2>&1 | grep -E "error TS" | head -20