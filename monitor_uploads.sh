#!/bin/bash

echo "🔍 Monitoring Upload Frequency (Press Ctrl+C to stop)"
echo ""
echo "With --neuron.upload_steps 3:"
echo "  ✅ Should upload every 3 training steps (~12-15 minutes)"
echo "  ✅ Old: Every 5 steps (~25 minutes)"
echo ""
echo "Recent uploads:"
echo "----------------------------------------"

pm2 logs distributed_training_miner --nostream --lines 500 | grep "Successfully pushed" | tail -10

echo ""
echo "----------------------------------------"
echo "Watching for new uploads (live)..."
echo ""

pm2 logs distributed_training_miner --lines 0 | grep --line-buffered -E "Successfully pushed|💾 Saving|🏋️"

