#!/bin/bash
FILE_ID="1-QoEGEOU5gQSxPpI01TXFuDV_iLTcM37"
OUTPUT_FILE="src/orca/checkpoints/tf_octa_c_ports.pth"

# Using curl with Google Drive URL format
curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o "${OUTPUT_FILE}"

echo "Downloaded checkpoint to ${OUTPUT_FILE}"