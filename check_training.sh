#!/bin/bash
# Quick script to check if training is active

echo "════════════════════════════════════════════════════"
echo "🔍 TRAINING STATUS CHECK"
echo "════════════════════════════════════════════════════"
echo ""

# Check PM2 status
echo "1️⃣  PM2 Process Status:"
pm2 status distributed_training_miner | grep distributed_training_miner
echo ""

# Check for recent training steps
echo "2️⃣  Recent Training Steps (last 5):"
pm2 logs distributed_training_miner --lines 2000 --nostream 2>/dev/null | \
  grep "🏋️.*Outer Step:" | tail -5
echo ""

# Check last activity timestamp
echo "3️⃣  Last Log Activity:"
pm2 logs distributed_training_miner --lines 50 --nostream 2>/dev/null | \
  grep -E "INFO|WARN|ERROR" | tail -1
echo ""

# Check for AllReduce requests
echo "4️⃣  AllReduce Requests (last 10):"
pm2 logs distributed_training_miner --lines 2000 --nostream 2>/dev/null | \
  grep -i "allreduce" | tail -10
if [ $? -ne 0 ]; then
  echo "   ❌ No AllReduce requests found"
fi
echo ""

# Check current epoch
echo "5️⃣  Current Epoch:"
pm2 logs distributed_training_miner --lines 500 --nostream 2>/dev/null | \
  grep "New Model Tag:" | tail -1
echo ""

echo "════════════════════════════════════════════════════"
echo "📊 INTERPRETATION:"
echo "════════════════════════════════════════════════════"
echo "✅ Training is ACTIVE if you see:"
echo "   - Recent 🏋️ Outer Step logs (every 5 minutes)"
echo "   - AllReduce requests being processed"
echo ""
echo "❌ Training is IDLE if:"
echo "   - No 🏋️ steps in last 10+ minutes"
echo "   - No AllReduce activity"
echo "════════════════════════════════════════════════════"
