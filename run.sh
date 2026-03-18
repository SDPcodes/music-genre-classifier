#!/bin/bash

# Exit immediately if any command fails
set -e

echo "------------------------------------------"
echo "Step 1: Merging datasets..."
python script/merge_dataset.py

echo "------------------------------------------"
echo "Step 2: Training Spark model..."
# Runs training and exits automatically due to sys.exit(0) in Scala
spark-shell --master "local[*]" -i spark/train.scala

echo "------------------------------------------"
echo "Step 3: Starting web server..."
# Open the browser automatically after a short delay (macOS)
(sleep 3 && open "http://127.0.0.1:8080") &

# Start the Flask server
python web/server.py