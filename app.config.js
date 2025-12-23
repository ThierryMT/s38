module.exports = {
  apps : [{
    name   : 'distributed_training_miner',
    script : 'neurons/miner.py',
    cwd: '/workspace/s38',
    interpreter: '/workspace/s38/.venv/bin/python',
    interpreter_args: '-m torch.distributed.run --nproc_per_node=4',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--netuid','38','--subtensor.chain_endpoint','wss://entrypoint-finney.opentensor.ai:443','--wallet.name','alexstev','--wallet.hotkey','newstart5','--axon.port','18515','--dht.port','20450','--dht.ip','154.42.3.50','--neuron.upload_steps','60','--neuron.local_batch_size_train','2','--neuron.local_batch_size_train_effective','32'],
    env: {
      R2_ACCOUNT_ID: '7cd6c1105bbf838dc5074cd338fd3628',
      R2_BUCKET_NAME: 'llama-4b-ws-4-227',
      R2_READ_ACCESS_KEY_ID: '62b80a6126b0346e2fb5b4449a7ee083',
      R2_READ_SECRET_ACCESS_KEY: '8608f43cd9910448897b90de3d3146044e39a69cc7c79e5636a1374586339ed5',
      R2_WRITE_ACCESS_KEY_ID: '2a1cfb823153758e2c786cca5048d1bb',
      R2_WRITE_SECRET_ACCESS_KEY: 'fb3adafe3c49a1c94316a6c5bbec1397ba0a3267a142c7348ccebe9089a447bb',
      R2_ADMIN_ACCESS_KEY_ID: '79e2cc3349038705923780abe1633368',
      R2_ADMIN_SECRET_ACCESS_KEY: '275002bc8c9f6963b0f8478e64dc679f0ac6f24999413cafdb536beee7fa3353',
      TORCH_DISTRIBUTED_INIT_TIMEOUT: '3600',
      NCCL_TIMEOUT: '3600',
      GLOO_SOCKET_IFNAME: 'lo',
      TORCHELASTIC_ERROR_FILE: '/tmp/torchelastic_error.json'
    }
  }]
}
