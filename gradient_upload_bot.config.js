module.exports = {
  apps : [{
    name   : 'gradient_upload_bot',
    script : 'gradient_upload_bot.py',
    interpreter: '/venv/main/bin/python3',
    min_uptime: '10s',
    max_restarts: '10',
    autorestart: true,
    watch: false,
    error_file: './logs/gradient_upload_bot_error.log',
    out_file: './logs/gradient_upload_bot_out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    env: {
      NODE_ENV: 'production'
    }
  }]
}

