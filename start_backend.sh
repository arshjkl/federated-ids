#!/bin/bash
cd /Users/admin/Desktop/federated-ids && python -m uvicorn src.dashboard.main:app --reload --port 18003