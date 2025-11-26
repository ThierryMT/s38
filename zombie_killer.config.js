module.exports = {
  apps : [{
    name   : 'zombie_killer',
    script : '/workspace/DistributedTraining/kill_zombie_processes.sh',
    interpreter: 'bash',
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    error_file: '/root/.pm2/logs/zombie-killer-error.log',
    out_file: '/root/.pm2/logs/zombie-killer-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    env: {
      LOOP_MODE: 'true',
      CHECK_INTERVAL: '30'
    }
  }]
}

