module.exports = {
  apps : [{
    name   : 'distributed_training_miner',
    script : 'neurons/miner.py',
    interpreter: '/venv/main/bin/torchrun',
    interpreter_args: '--nproc_per_node=' + '4',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--netuid','38','--subtensor.chain_endpoint','wss://entrypoint-finney.opentensor.ai:443','--wallet.name','alexstev','--wallet.hotkey','newstart17','--axon.port','19644','--dht.port','19944','--dht.ip','154.42.3.69','--neuron.upload_steps','30','--neuron.local_batch_size_train','2','--neuron.local_batch_size_train_effective','32']
  }]
}
