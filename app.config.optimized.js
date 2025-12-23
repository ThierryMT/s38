// Optimized PM2 Configuration for Maximum Assigned Score
// Based on scoring analysis from distributed_training/validator/reward.py

module.exports = {
  apps: [{
    name: 'distributed_training_miner',
    script: 'neurons/miner.py',
    interpreter: '/venv/main/bin/torchrun',
    interpreter_args: '--nproc_per_node=' + '4',
    min_uptime: '5m',
    max_restarts: '5',
    
    args: [
      // Network Configuration
      '--netuid', '38',
      '--subtensor.chain_endpoint', 'wss://entrypoint-finney.opentensor.ai:443',
      '--wallet.name', 'alexstev',
      '--wallet.hotkey', 'newstart5',
      '--axon.port', '13330',
      '--dht.port', '19944',
      '--dht.ip', '154.42.3.69',
      
      // ⭐ CRITICAL SCORING PARAMETERS ⭐
      
      // Upload frequency: More frequent uploads = better R2 validity score
      // Current: 120, Recommended: 60-80 for better repo freshness
      '--neuron.upload_steps', '60',  // CHANGED: Upload 2x more frequently
      
      // Batch size configuration
      // Keep these aligned with your GPU memory (4 GPUs with ~80GB VRAM total)
      '--neuron.local_batch_size_train', '2',  // Per-GPU batch size (OK for memory)
      
      // ⭐ CRITICAL: Total effective batch across all GPUs and accumulation steps
      // Formula: local_batch_size_train × accumulation_steps × num_gpus
      // Current: 512 = 2 × 64 × 4 (64 accumulation steps)
      // Recommendation: Increase to 1024 for better gradient quality
      '--neuron.local_batch_size_train_effective', '1024',  // CHANGED: Better gradient quality
      
      // ⭐ Learning rate warmup and scheduling
      // Higher learning rate = better loss improvement = higher assigned_score
      // The miner code already uses 3.0e-4 (higher than validator's 2.5e-4) - GOOD!
      
      // ⭐ Training steps before upload
      // More inner steps = better gradient quality before upload
      // Miner code uses 700 (validator uses 500) - ALREADY OPTIMIZED!
      
      // Additional optimization: Enable all rank logs for debugging
      '--show_all_rank_logs'
    ],
    
    // Environment variables for distributed training
    env: {
      // R2 Storage credentials (required for repo_valid_score)
      R2_ACCOUNT_ID: '7cd6c1105bbf838dc5074cd338fd3628',
      R2_BUCKET_NAME: 'llama-4b-ws-4-227',
      R2_READ_ACCESS_KEY_ID: '62b80a6126b0346e2fb5b4449a7ee083',
      R2_READ_SECRET_ACCESS_KEY: '8608f43cd9910448897b90de3d3146044e39a69cc7c79e5636a1374586339ed5',
      R2_WRITE_ACCESS_KEY_ID: '2a1cfb823153758e2c786cca5048d1bb',
      R2_WRITE_SECRET_ACCESS_KEY: 'fb3adafe3c49a1c94316a6c5bbec1397ba0a3267a142c7348ccebe9089a447bb',
      R2_ADMIN_ACCESS_KEY_ID: '79e2cc3349038705923780abe1633368',
      R2_ADMIN_SECRET_ACCESS_KEY: '275002bc8c9f6963b0f8478e64dc679f0ac6f24999413cafdb536beee7fa3353',
      
      // Distributed training timeouts (increased for stability)
      TORCH_DISTRIBUTED_INIT_TIMEOUT: '3600',  // 1 hour
      NCCL_TIMEOUT: '3600',                     // 1 hour
      GLOO_SOCKET_IFNAME: 'lo',                 // Use localhost
      
      // ⭐ PyTorch optimizations for better training performance
      TORCH_CUDNN_BENCHMARK: '1',              // Enable cuDNN auto-tuner
      PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True',  // Better memory management
      
      // Logging
      PYTHONUNBUFFERED: '1',                    // Immediate log output
      WANDB_SILENT: 'true'                      // Reduce wandb noise in logs
    }
  }]
}

