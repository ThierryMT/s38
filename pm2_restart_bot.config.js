// PM2 configuration for the auto-restart bot
module.exports = {
  apps: [{
    name: 'pm2_restart_bot',
    script: './pm2_auto_restart_bot.py',
    interpreter: 'python3',
    autorestart: true,
    watch: false,
    max_memory_restart: '500M',
    error_file: './logs/pm2_restart_bot_error.log',
    out_file: './logs/pm2_restart_bot_out.log',
    log_file: './logs/pm2_restart_bot_combined.log',
    time: true,
    env: {
      NODE_ENV: 'production'
    }
  }]
};

