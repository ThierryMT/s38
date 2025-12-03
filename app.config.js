module.exports = {
  apps : [{
    name   : 'distributed_training_miner',
    script : 'neurons/miner.py',
    interpreter: 'torchrun',
    interpreter_args: '--nproc_per_node=' + '4',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--netuid','38','--subtensor.chain_endpoint','wss://entrypoint-finney.opentensor.ai:443','--wallet.name','alexstev','--wallet.hotkey','newstart5','--axon.port','25466','--dht.port','24376','--dht.ip','154.42.3.51','--neuron.upload_steps','30','--neuron.local_batch_size_train','2','--neuron.local_batch_size_train_effective','32']
  }]
}
