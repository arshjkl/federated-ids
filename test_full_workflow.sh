#!/bin/bash
echo "1. Analyzing sample dataset..."
ANALYSIS=$(curl -s -X POST http://localhost:18003/api/inference/analyze-sample \
  -F 'sample_file=mock_inference_batch_01.csv')
DATASET_ID=$(echo $ANALYSIS | jq -r '.dataset_id')
echo "Dataset ID: $DATASET_ID"

echo "2. Validating dataset..."
curl -s http://localhost:18003/api/inference/validate/$DATASET_ID | jq .

echo "3. Preprocessing dataset..."
curl -s -X POST http://localhost:18003/api/inference/preprocess/$DATASET_ID | jq .

echo "4. Running inference..."
MODEL_ID="99dccf30-5321-4b6e-8373-7efd8058c5ab"
curl -s -X POST http://localhost:18003/api/inference/upload-sample \
  -F "model_id=$MODEL_ID" \
  -F 'name=Test Inference' \
  -F 'sample_file=mock_inference_batch_01.csv' \
  -F 'threshold=0.5' \
  -F 'batch_size=256' | jq .