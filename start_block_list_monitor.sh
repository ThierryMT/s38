#!/bin/bash
# Start the Block List Monitor Bot

echo "Starting Block List Monitor Bot..."

# Make the Python script executable
chmod +x block_list_monitor_bot.py

# Create logs directory if it doesn't exist
mkdir -p logs

# Option 1: Run directly in terminal (you can see live output)
# python3 block_list_monitor_bot.py

# Option 2: Run as a pm2 process (recommended for production)
pm2 start block_list_monitor_bot.config.js

echo "Bot started! Check status with: pm2 status block_list_monitor_bot"
echo "View logs with: pm2 logs block_list_monitor_bot"
echo "Stop bot with: pm2 stop block_list_monitor_bot"

