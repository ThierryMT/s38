#!/bin/bash
# Start the PM2 Auto-Restart Bot

echo "Starting PM2 Auto-Restart Bot..."

# Make the Python script executable
chmod +x pm2_auto_restart_bot.py

# Option 1: Run directly in terminal (you can see live output)
# python3 pm2_auto_restart_bot.py

# Option 2: Run as a pm2 process (recommended for production)
pm2 start pm2_restart_bot.config.js

echo "Bot started! Check status with: pm2 status pm2_restart_bot"
echo "View logs with: pm2 logs pm2_restart_bot"

