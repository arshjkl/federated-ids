#!/bin/bash
JOB_ID="${1:-fd45b06a-7e15-4fb9-b25d-8b4f1c078f75}"
curl -s "http://localhost:18003/api/inference/$JOB_ID/results" | jq .
