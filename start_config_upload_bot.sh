#!/bin/bash
# Start the Config Upload Bot

echo "Starting Config Upload Bot..."

# Make the Python script executable
chmod +x config_upload_bot.py

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the bot with PM2
pm2 start config_upload_bot.config.js

echo ""
echo "✅ Bot started!"
echo "📊 Check status with: pm2 status config_upload_bot"
echo "📋 View logs with: pm2 logs config_upload_bot"
echo "⏸️  Stop bot with: pm2 stop config_upload_bot"
echo "🔄 Restart bot with: pm2 restart config_upload_bot"

