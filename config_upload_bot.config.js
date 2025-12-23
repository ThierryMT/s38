// PM2 Configuration for Config Upload Bot
// Uploads config.json every 30 seconds to keep block_list fresh for validators
module.exports = {
  apps: [{
    name: 'config_upload_bot',
    script: './config_upload_bot.py',
    interpreter: '/workspace/s38/.venv/bin/python3',
    autorestart: true,
    watch: false,
    max_memory_restart: '100M',
    error_file: './logs/config_upload_bot_error.log',
    out_file: './logs/config_upload_bot_out.log',
    log_file: './logs/config_upload_bot_combined.log',
    time: true,
    env: {
      NODE_ENV: 'production'
    }
  }]
};
