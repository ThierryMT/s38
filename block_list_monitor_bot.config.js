// PM2 configuration for the block list monitor bot
// NOTE: When the bot restarts the distributed_training_miner process,
// it will FIRST clear all GPU processes to free up GPU memory before restarting.
// This prevents CUDA out of memory errors.
module.exports = {
  apps: [{
    name: 'block_list_monitor_bot',
    script: './block_list_monitor_bot.py',
    interpreter: 'python3',
    autorestart: true,
    watch: false,
    max_memory_restart: '200M',
    error_file: './logs/block_list_monitor_bot_error.log',
    out_file: './logs/block_list_monitor_bot_out.log',
    log_file: './logs/block_list_monitor_bot_combined.log',
    time: true,
    env: {
      NODE_ENV: 'production'
    }
  }]
};

