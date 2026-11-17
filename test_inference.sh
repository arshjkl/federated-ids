#!/bin/bash
curl -s -X POST http://localhost:8000/api/inference/upload-sample \
  -F 'model_id=99dccf30-5321-4b6e-8373-7efd8058c5ab' \
  -F 'name=Test Inference' \
  -F 'sample_file=mock_inference_batch_01.csv' \
  -F 'threshold=0.5' \
  -F 'batch_size=256' | jq .