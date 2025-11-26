module.exports = {
  apps : [{
    name   : 'distributed_training_miner',
    script : 'neurons/miner.py',
    interpreter: 'torchrun',
    interpreter_args: '--nproc_per_node=' + '4',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--netuid','38','--subtensor.chain_endpoint','wss://entrypoint-finney.opentensor.ai:443','--wallet.name','alexstev','--wallet.hotkey','newstart3','--axon.port','31917','--dht.port','31742','--dht.ip','49.213.134.9','--neuron.upload_steps','3']
  }]
}
