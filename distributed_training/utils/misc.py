# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import time
import socket
import threading
from functools import lru_cache, update_wrapper
from ipaddress import ip_address
from math import floor
from typing import Any, Callable

import bittensor as bt
import hivemind
import speedtest
from hivemind import utils

import wandb
from distributed_training import __run__, __version__
from copy import deepcopy
from dataclasses import is_dataclass, asdict


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(self)

    Note: self here is the miner or validator instance
    """
    return self.subtensor.get_current_block()


def to_plain_dict(obj):
    if isinstance(obj, dict):
        return deepcopy(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # pydantic v1
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return deepcopy(getattr(obj, "__dict__", {}))


def sanitize_wandb_config(cfg):
    cfg_dict = to_plain_dict(cfg)
    # remove the entire sensitive subtree
    cfg_dict.pop("r2", None)
    return cfg_dict


def load_wandb(self, config, wallet, neuron_type, peer_id):
    run_name = f"{neuron_type[0].upper()}{'{:03}'.format(self.uid)}"

    tags = [peer_id, __version__, self.wallet.hotkey.ss58_address, f"run{__run__}"]

    run_id = "_".join([run_name] + tags[2:]).lower()

    wandb_run = wandb.init(
        id=run_id,
        name=run_name,
        anonymous="allow",
        resume="allow",
        tags=tags,
        project=config.neuron.wandb_project,
        entity=config.neuron.wandb_entity,
        config={},
        allow_val_change=True,
    )

    sanitized_config = sanitize_wandb_config(config)
    wandb_run.config.update(sanitized_config, allow_val_change=True)

    return wandb_run


def get_bandwidth():
    # Get speedtest results
    s = speedtest.Speedtest()
    s.get_servers()
    s.get_best_server()
    s.download()
    s.upload()
    results = s.results.dict()

    # Copy key metrics to a formatted badnwidth_dict
    bandwidth_dict = {}
    keys = ["download", "upload", "ping"]
    for key in keys:
        bandwidth_dict[f"all_reduce/{key}"] = float(f"{results[key] / 1e6:.2f}")

    return bandwidth_dict


def init_dht(self):
    if self.master:
        # Init DHT and model
        if self.config.dht.ip:
            version = "4"
            address = self.config.dht.ip
            announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]
        else:
            address = bt.utils.networking.get_external_ip()
            self.logger.info(f"Received public IP address of this machine: {address}")
            version = ip_address(address).version
            announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]

        # Init list of available DHT addresses from wandb
        initial_peers_list = self.config.neuron.initial_peers.copy()
        
        def fetch_wandb_peers():
            """Fetch DHT peers from WandB in a separate thread with timeout protection"""
            try:
                self.logger.info("Fetching DHT addresses from WandB...")
                
                # Note: signal.alarm doesn't work in threads, but the 15-second thread timeout handles it
                api = wandb.Api(timeout=10)  # Set API timeout
                
                # Fetch validator runs
                self.logger.info("Fetching validator runs from WandB...")
                validator_runs = api.runs(
                    f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project.replace('_validators','').replace('_miners','')}_validators"
                )
                for ru in validator_runs:
                    if ru.state == "running":
                        if "dht_addresses" in ru.config.get("neuron", {}).keys():
                            for peer in ru.config["neuron"]["dht_addresses"]:
                                if peer not in initial_peers_list:
                                    initial_peers_list.append(peer)

                # Fetch miner runs
                self.logger.info("Fetching miner runs from WandB...")
                miner_runs = api.runs(
                    f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project.replace('_validators','').replace('_miners','')}_miners"
                )
                for ru in miner_runs:
                    if ru.state == "running":
                        if "dht_addresses" in ru.config.get("neuron", {}).keys():
                            for peer in ru.config["neuron"]["dht_addresses"]:
                                if peer not in initial_peers_list:
                                    initial_peers_list.append(peer)
                self.logger.info(f"Successfully fetched DHT addresses from WandB. Total peers: {len(initial_peers_list)}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch DHT addresses from WandB: {e}")
        
        # Try to fetch from WandB with 15 second timeout
        self.logger.info("Starting WandB fetch thread with 15s timeout...")
        wandb_thread = threading.Thread(target=fetch_wandb_peers, daemon=True)
        wandb_thread.start()
        wandb_thread.join(timeout=15.0)
        
        if wandb_thread.is_alive():
            self.logger.warning("WandB fetch timed out after 15 seconds. Using only initial_peers from config.")
        
        self.logger.info(f"Proceeding with {len(initial_peers_list)} DHT peers")

        # Init DHT with smarter retry logic
        retries = 0
        max_retries_per_peer = 2  # Only try each peer twice
        successful_connection = False
        failed_peers = set()
        
        # Try each peer, but skip ones that have already failed
        while not successful_connection and retries < (len(initial_peers_list) * max_retries_per_peer):
            for initial_peer in initial_peers_list:
                if successful_connection:
                    break
                    
                # Skip peers that have consistently failed
                if initial_peer in failed_peers:
                    continue
                    
                try:
                    self.logger.info(f"Attempting DHT connection to {initial_peer} (attempt {retries + 1})")
                    
                    # Use threading to add timeout to DHT initialization
                    dht_result = {'dht': None, 'error': None}
                    
                    def init_dht_with_peer():
                        try:
                            dht_result['dht'] = hivemind.DHT(
                                host_maddrs=[
                                    f"/ip4/0.0.0.0/tcp/{self.config.dht.port}",
                                    f"/ip4/0.0.0.0/udp/{self.config.dht.port}/quic",
                                ],
                                initial_peers=[initial_peer],
                                announce_maddrs=announce_maddrs,
                                start=True,
                            )
                        except Exception as e:
                            dht_result['error'] = e
                    
                    dht_thread = threading.Thread(target=init_dht_with_peer, daemon=True)
                    dht_thread.start()
                    dht_thread.join(timeout=30)  # 30 second timeout per attempt
                    
                    if dht_thread.is_alive():
                        # Thread is still running - timeout occurred
                        raise Exception(f"DHT initialization timed out after 30 seconds")
                    
                    if dht_result['error']:
                        raise dht_result['error']
                    
                    if dht_result['dht']:
                        self.dht = dht_result['dht']
                        self.logger.info(
                            f"✅ Successfully initialised DHT using initial_peer as {initial_peer}"
                        )
                        successful_connection = True
                        utils.log_visible_maddrs(
                            self.dht.get_visible_maddrs(), only_p2p=True
                        )
                        # Add DHT address to wandb config
                        self.config.neuron.dht_addresses = [
                            re.sub(
                                "ip4/?(.*?)/",
                                f"ip{version}/{address}/",
                                str(addr),
                                flags=re.DOTALL,
                            )
                            for addr in self.dht.get_visible_maddrs()
                        ]
                        return
                    else:
                        raise Exception("DHT initialization returned None")
                        
                except Exception as e:
                    self.logger.warning(
                        f"❌ DHT connection to {initial_peer} failed: {str(e)[:100]}"
                    )
                    retries += 1
                    
                    # After 2 failures on this peer, mark it as failed
                    peer_fail_count = sum(1 for _ in range(retries) if _ % len(initial_peers_list) == initial_peers_list.index(initial_peer))
                    if peer_fail_count >= max_retries_per_peer:
                        failed_peers.add(initial_peer)
                        self.logger.warning(f"Marking peer {initial_peer} as failed after {max_retries_per_peer} attempts")
                    
                    if not successful_connection:
                        time.sleep(2)  # Shorter sleep
        
        # If all peers failed, raise exception
        if not successful_connection:
            self.logger.error(f"Failed to connect to any of {len(initial_peers_list)} DHT peers after {retries} total attempts")
            raise Exception(f"DHT initialization failed: All {len(initial_peers_list)} peers unreachable")


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is available for binding.
    
    Args:
        port (int): The port number to check.
        host (str): The host address to check. Defaults to "127.0.0.1".
    
    Returns:
        bool: True if the port is available, False otherwise.
    
    This function is useful for checking if the distributed training port (e.g., 29500)
    is available before attempting to initialize the process group.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except socket.error:
        sock.close()
        return False


def wait_for_port_available(port: int, host: str = "127.0.0.1", timeout: int = 30, logger=None) -> bool:
    """
    Wait for a port to become available with timeout.
    
    Args:
        port (int): The port number to wait for.
        host (str): The host address. Defaults to "127.0.0.1".
        timeout (int): Maximum seconds to wait. Defaults to 30.
        logger: Optional logger instance for logging.
    
    Returns:
        bool: True if port became available, False if timeout reached.
    
    This function polls the port status and waits for it to become available,
    which is useful when restarting processes that need to bind to the same port.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_port_available(port, host):
            if logger:
                logger.info(f"Port {port} is now available.")
            return True
        if logger:
            logger.debug(f"Port {port} still in use, waiting...")
        time.sleep(1)
    
    if logger:
        logger.error(f"Timeout waiting for port {port} to become available after {timeout}s.")
    return False
