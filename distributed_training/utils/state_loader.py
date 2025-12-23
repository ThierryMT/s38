import copy
import hivemind
import bittensor as bt
import logging
import json
import gc
import os
import shutil
import psutil
import torch
import time

from functools import partial
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from distributed_training import __run__
from distributed_training.averaging.averagers import DTGradAverager
from distributed_training.utils.progress_tracker import (
    get_progress,
    get_r2_client,
)
from distributed_training.utils.r2 import (
    upload_folder_to_r2,
    r2_download,
    log_peerid_to_r2,
)
from distributed_training.averaging.avg_handler import AveragingHandler
from torch.distributed._tensor import DeviceMesh
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_optimizer_state_dict,
    set_model_state_dict,
)
import torch.distributed as dist
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from itertools import chain
from hivemind.utils import (
    get_logger,
    nested_flatten,
)
from packaging.version import Version
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh as DM
from torch.distributed.tensor.placement_types import Shard
from safetensors.torch import save_file, load_file
from botocore.client import BaseClient

hivemind_logger = get_logger(__name__)

torch.serialization.add_safe_globals([DTensorSpec])
torch.serialization.add_safe_globals([DTensor])
torch.serialization.add_safe_globals([DM])
torch.serialization.add_safe_globals([Shard])
torch.serialization.add_safe_globals([TensorMeta])

Parameters = Iterable[torch.Tensor]
ParamGroups = Iterable[Dict[str, Any]]
TorchOptimizer = torch.optim.Optimizer
if Version(torch.__version__).major >= 2:
    ZERO_GRAD_SET_TO_NONE_DEFAULT = True
    LRSchedulerBase = torch.optim.lr_scheduler.LRScheduler
else:
    ZERO_GRAD_SET_TO_NONE_DEFAULT = False
    LRSchedulerBase = torch.optim.lr_scheduler._LRScheduler
OptimizerFactory = Callable[[Union[Parameters, ParamGroups]], TorchOptimizer]
SchedulerFactory = Callable[[TorchOptimizer], LRSchedulerBase]


@staticmethod
def check_params(
    optimizer: Union[TorchOptimizer, OptimizerFactory],
    param_groups: Optional[Union[Parameters, ParamGroups]],
    parameter_names: Optional[Sequence[str]],
) -> Tuple[ParamGroups, Sequence[torch.Tensor], Sequence[str]]:
    """Get and verify parameters, groups and names"""
    if param_groups is None:
        assert hasattr(
            optimizer, "param_groups"
        ), "Must provide param_groups or an optimizer with .param_groups"
        param_groups = optimizer.param_groups
    param_groups = tuple(param_groups)
    if all(isinstance(p, torch.Tensor) for p in param_groups):
        param_groups = (dict(params=param_groups),)
    for group in param_groups:
        assert isinstance(group, dict) and group.get("params") is not None
        assert all(isinstance(p, torch.Tensor) for p in group["params"])
    parameters = tuple(chain(*(group["params"] for group in param_groups)))
    if parameter_names is None:
        parameter_names = tuple(i for i in range(len(parameters)))
    parameter_names = tuple(nested_flatten(parameter_names))
    assert len(parameters) == len(
        parameter_names
    ), f"Expected {len(parameters)} names, got {len(parameter_names)}"
    assert len(set(parameters)) == len(
        parameters
    ), "Found duplicate parameters in param_groups"
    params_with_grad = sum(p.numel() for p in parameters if p.requires_grad)
    params_no_grad = sum(p.numel() for p in parameters if not p.requires_grad)
    if params_no_grad >= params_with_grad:
        bt.logging.info(
            "The majority of parameters have requires_grad=False, but they are still synchronized"
            " with peers. If these parameters are frozen (not updated), please do not feed them into "
            "the optimizer at all in order to avoid communication overhead. Proceeding anyway."
        )

    return param_groups, parameters, parameter_names


def make_averaged_parameters(self, main_parameters: Sequence[torch.Tensor]):
    """Initialize averaged parameters based on the optimizer and averaging mode"""
    return tuple(
        make_host_tensor(param, force_copy=self.offload_optimizer)
        for param in main_parameters
    )


def make_host_tensor(
    source_tensor: torch.Tensor, reuse_tensors: bool = False, force_copy: bool = False
) -> torch.Tensor:
    """Create a new tensor for averaging or reuse the existing one"""
    if reuse_tensors and not force_copy:
        if source_tensor.device != torch.device("cpu"):
            raise ValueError(
                "reuse_tensors is only supported if all averaged tensors are on CPU"
            )
        if not source_tensor.is_shared():
            source_tensor.share_memory_()
        return source_tensor
    else:
        averaged_tensor = source_tensor.detach().to(
            device="cpu", dtype=torch.float32, copy=True
        )
        return averaged_tensor.share_memory_().requires_grad_(
            source_tensor.requires_grad
        )


def init_components(
    self,
    main_parameters,
    param_groups: ParamGroups,
    optimizer_or_factory: Union[TorchOptimizer, OptimizerFactory],
    scheduler_or_factory: Optional[Union[LRSchedulerBase, SchedulerFactory]],
    initialize_optimizer: Optional[bool],
) -> Tuple[TorchOptimizer, Optional[LRSchedulerBase]]:
    """Get optimizer and scheduler by either instantiating user-provided factory or using pre-instantiated ones"""
    # assert hasattr(self, "_averaged_parameters"), "Internal error: must initialize averaged parameters first"
    optimizer_is_factory = callable(optimizer_or_factory) and not isinstance(
        optimizer_or_factory, TorchOptimizer
    )
    scheduler_is_factory = callable(scheduler_or_factory) and not isinstance(
        scheduler_or_factory, LRSchedulerBase
    )
    if (
        optimizer_is_factory
        and not scheduler_is_factory
        and scheduler_or_factory is not None
    ):
        raise ValueError(
            "If optimizer is created internally, scheduler must also be initialized internally"
        )
    if self.offload_optimizer and not optimizer_is_factory:
        raise ValueError(
            "Using offload_optimizer requires creating optimizer inside hivemind"
        )
    # self.logger.info("Before make_averaged_parameters")
    averaged_parameters = make_averaged_parameters(self, main_parameters)

    # create optimizer
    if optimizer_is_factory:
        if self.offload_optimizer:
            if self.reuse_tensors:
                parameters_for_optimizer = averaged_parameters
            else:
                parameters_for_optimizer = tuple(
                    tensor.detach().clone().requires_grad_(tensor.requires_grad)
                    for tensor in averaged_parameters
                )

            next_index = 0
            param_groups_for_optimizer = []
            for param_group in param_groups:
                num_params = len(param_group["params"])
                averaged_params_for_group = parameters_for_optimizer[
                    next_index : next_index + num_params
                ]
                param_groups_for_optimizer.append(
                    dict(param_group, params=averaged_params_for_group)
                )
                next_index += num_params
            assert next_index == len(parameters_for_optimizer)

            for param in parameters_for_optimizer:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
        else:
            param_groups_for_optimizer = param_groups
        optimizer = optimizer_or_factory(param_groups_for_optimizer)
    else:
        optimizer = optimizer_or_factory

    # optionally initialize optimizer state dict
    if initialize_optimizer is None:
        initialize_optimizer = not any(
            isinstance(x, torch.Tensor) for x in nested_flatten(optimizer.state_dict())
        )
        bt.logger.info(
            self.status_loglevel,
            "Initializing optimizer manually since it has no tensors in state dict. "
            "To override this, provide initialize_optimizer=False",
        )

    if initialize_optimizer:
        initialize_optimizer_state_(
            optimizer
        )  # note: this will run one optimizer step!

    # create LR scheduler
    if scheduler_is_factory:
        assert callable(scheduler_or_factory)
        scheduler = scheduler_or_factory(optimizer)
    else:
        scheduler = scheduler_or_factory

    # verify optimizer and scheduler
    assert isinstance(optimizer, TorchOptimizer) and len(optimizer.param_groups) == len(
        list(param_groups)
    )
    if self.reuse_tensors:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                assert param.is_shared()
    assert isinstance(scheduler, (LRSchedulerBase, type(None)))
    if scheduler is not None:
        assert scheduler.optimizer == optimizer
    return optimizer, scheduler


def initialize_optimizer_state_(opt: torch.optim.Optimizer):
    """Initialize optimizer statistics by running a virtual optimizer step with zero gradients"""
    flat_params = tuple(
        param for group in opt.param_groups for param in group["params"]
    )
    old_grads = []
    for param in flat_params:
        old_grads.append(param.grad)
        param.grad = torch.zeros_like(param)
    opt.step()
    for param, old_grad in zip(flat_params, old_grads):
        param.grad = old_grad


def check_model_exists(
    self,
    r2: BaseClient,
    bucket_name: str,
    prefix: str = "",
    revision: Optional[str] = None,
) -> bool:
    try:
        obj = r2.get_object(Bucket=bucket_name, Key=f"{prefix}metadata.json")
        data = obj["Body"].read()
        metadata = json.loads(data)
        metadata_revision = (
            f"{metadata['run']}.{metadata['outer_step']}.{metadata['inner_step']}"
        )
        if revision and revision != "None" and revision != metadata_revision:
            return False
        else:
            return True

    except Exception as e:
        self.logger.info(f"Model or revision check failed with error: {e}")
        return False


def check_epoch_model_exists_with_retry(
    self,
    r2: BaseClient,
    bucket_name: str,
    epoch: int,
    max_wait_minutes: int = 3,
) -> bool:
    """
    Check if an epoch model exists in R2 with exponential backoff retry logic.
    This is specifically designed to handle epoch transitions gracefully.
    OPTIMIZED: Now defaults to 3 minutes with faster retries before falling back.
    
    Args:
        r2: R2 client
        bucket_name: Bucket name in R2
        epoch: Epoch number to check
        max_wait_minutes: Maximum minutes to wait for epoch model (default: 3)
    
    Returns:
        True if model exists, False if max wait time exceeded
    """
    import time
    
    prefix = f"epoch-{epoch}/"
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    attempt = 0
    base_wait = 5  # Start with 5 second wait (reduced from 10)
    max_single_wait = 30  # Cap individual waits at 30 seconds (reduced from 180)
    
    self.logger.info(f"🔍 Checking for epoch-{epoch} model in {bucket_name}...")
    
    while (time.time() - start_time) < max_wait_seconds:
        try:
            obj = r2.get_object(Bucket=bucket_name, Key=f"{prefix}metadata.json")
            self.logger.info(f"✅ Epoch-{epoch} model found in R2!")
            return True
        except Exception as e:
            attempt += 1
            elapsed = int(time.time() - start_time)
            
            # Check if it's a 404 error (model doesn't exist yet)
            error_str = str(e)
            is_not_found = "404" in error_str or "NoSuchKey" in error_str or "Not Found" in error_str
            
            if is_not_found:
                # Calculate exponential backoff wait time
                wait_time = min(base_wait * (2 ** (attempt - 1)), max_single_wait)
                remaining = max_wait_seconds - elapsed
                
                if remaining <= 0:
                    self.logger.error(
                        f"❌ Timeout: Epoch-{epoch} model not available after {max_wait_minutes} minutes"
                    )
                    return False
                
                # Adjust wait time if it exceeds remaining time
                wait_time = min(wait_time, remaining)
                
                self.logger.warning(
                    f"⏳ Epoch-{epoch} model not available yet (attempt {attempt}). "
                    f"Waiting {int(wait_time)}s before retry... "
                    f"(elapsed: {elapsed}s/{max_wait_seconds}s) - Will try fallback after timeout"
                )
                time.sleep(wait_time)
            else:
                # Non-404 error, log and retry with shorter wait
                self.logger.error(f"❌ Error checking epoch-{epoch} model: {e}")
                time.sleep(5)
    
    self.logger.error(
        f"❌ Max wait time ({max_wait_minutes} minutes) exceeded for epoch-{epoch} model"
    )
    return False


def _bytes(x):
    return x.numel() * x.element_size()


def summarize_optimizer_state(opt, logger, tag):
    by_dtype = {}
    total = 0
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                nb = _bytes(v)
                total += nb
                key = (str(v.device), str(v.dtype))
                by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] OPT-state total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1])[:12]:
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def summarize_moments_only(opt, logger, tag):
    by_dtype = {}
    total = 0
    for st in opt.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            t = st.get(name, None)
            if isinstance(t, torch.Tensor):
                nb = _bytes(t)
                total += nb
                key = (str(t.device), str(t.dtype))
                by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] MOMENTS total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1]):
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def summarize_grads(model, logger, tag):
    by_dtype = {}
    total = 0
    for p in model.parameters():
        g = p.grad
        if isinstance(g, torch.Tensor):
            nb = _bytes(g)
            total += nb
            key = (str(g.device), str(g.dtype))
            by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] GRADS total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1]):
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def summarize_params(model, logger, tag):
    by_dtype = {}
    total = 0
    for p in model.parameters():
        t = p
        nb = _bytes(t)
        total += nb
        key = (str(t.device), str(t.dtype))
        by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] PARAMS total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1]):
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def cuda_mem(logger, tag):
    torch.cuda.synchronize()
    logger.info(
        f"[{tag}] alloc={torch.cuda.memory_allocated()/1e9:.3f} GB | "
        f"reserved={torch.cuda.memory_reserved()/1e9:.3f} GB"
    )


def check_cache_sync(self, r2, local_model_name, epoch, output_dir):
    """
    Smart cache validation that avoids unnecessary full redownloads during epoch transitions.
    
    Returns:
        True: All files match, use full cache
        False: Need to download (will download only what's needed based on what exists)
    """
    try:
        local_output_dir = os.path.join(os.getcwd(), local_model_name)
        metadata_file_path = os.path.join(local_output_dir, "metadata.json")
        
        # Check if metadata exists locally
        if not os.path.exists(metadata_file_path):
            try:
                r2_download(
                    self,
                    r2=r2,
                    bucket=local_model_name,
                    key=f"epoch-{epoch}/metadata.json",
                    donwload_on_all_ranks=False,
                    run_on_all_ranks=True,
                    destination=output_dir,
                )
            except Exception as e:
                self.logger.info(f"No metadata in R2 for epoch {epoch}: {e}")
                return False
        
        with open(metadata_file_path, "r") as file:
            metadata = json.load(file)
        
        # Check if this is exact same state (epoch AND inner_step match)
        if (self.local_progress.epoch, self.local_progress.inner_step) == (
            metadata["outer_step"],
            metadata["inner_step"],
        ):
            # Exact match - verify all files present
            files = os.listdir(local_output_dir)
            if local_model_name == self.config.neuron.global_model_name:
                required_files = [
                    "config.json",
                    "model.safetensors",
                    "outer_optimizer.pt",
                    "metadata.json",
                ]
            else:
                required_files = ["config.json", "model.safetensors", "metadata.json"]
            for i in range(self.world_size):
                required_files.append(
                    f"inner_optimizer.rank{i+1:04d}-of-{self.world_size}.pt"
                )
            if set(required_files).issubset(files):
                self.logger.info("✅ Skipping Download - Using Full Local Cache")
                return True
            else:
                self.logger.info(f"⚠️ Missing files, will redownload")
                return False
        else:
            # Different epoch/step - but check if we have reusable files
            files = os.listdir(local_output_dir) if os.path.exists(local_output_dir) else []
            
            # Check if config.json exists (model architecture - doesn't change between epochs)
            has_config = "config.json" in files
            
            if has_config:
                self.logger.info(
                    f"📦 Epoch transition detected ({self.local_progress.epoch} → {metadata['outer_step']}). "
                    f"Config exists locally - will download only weights and optimizer states"
                )
            else:
                self.logger.info(
                    f"📥 Full download needed for epoch {metadata['outer_step']} (no local config)"
                )
            
            # Return False to trigger download, but the download logic will skip config.json if it exists
            return False
            
    except Exception as e:
        self.logger.info(f"⚠️ Error checking cache: {e}. Will attempt download.")
        return False


def load_model_optimizer_gradient_averager(
    self,
    uid,
    epoch,
    reload_inner_optimizer=True,
    reload_outer_optimizer=True,
    revision=None,
    reset_block_list=True,
):
    """
    Pytorch currently have an ongoing issue with memory leaks:
    https://github.com/pytorch/pytorch/issues/64043. To mitigate
    against this for now gc.collect() is run after each component
    with optimizers and state averagers are deleted.
    """
    self.logger.info(f"Before Loading State {self.print_memory_usage()}")
    r2 = get_r2_client(self, uid, donwload_on_all_ranks=True)

    global_model_revision = f"{__run__}.{epoch}.0"
    global_model_name = self.config.neuron.global_model_name

    metadata_epoch = get_progress(self, "local", uid=uid)[0]
    if (revision is None) and (uid != self.master_uid):
        revision = f"{__run__}.{epoch}.{self.local_progress.inner_step}"
    elif (revision is None) and (uid == self.master_uid):
        revision = global_model_revision
    if epoch == metadata_epoch:
        prefix = ""
    else:
        prefix = f"epoch-{epoch}/"
    prefix = f"epoch-{epoch}/"

    local_model_name = (
        f"{self.config.neuron.global_model_name.split('/')[-1]}-{uid:03d}"
        if uid != self.master_uid
        else self.config.neuron.global_model_name
    )
    output_dir = os.path.join(os.getcwd(), local_model_name)
    global_output_dir = os.path.join(os.getcwd(), global_model_name)
    use_cache = check_cache_sync(self, r2, local_model_name, prefix, output_dir)
    use_global_cache = check_cache_sync(
        self, r2, global_model_name, prefix, global_output_dir
    )

    # Delete existing average handler
    if hasattr(self, "avg_handler"):
        del self.avg_handler.model
        del self.avg_handler.inner_optimizer
        del self.avg_handler.grad_averager
        del self.avg_handler
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.logger.info("Deleted Average Handler")

    if hasattr(self, "inner_optimizer"):
        for group in self.inner_optimizer.param_groups:
            group["params"].clear()
        self.inner_optimizer.state.clear()
        del self.inner_optimizer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    loading_success = 0
    optimizer_state = None
    # Load Model & Inner Optimizer
    try:
        if local_model_name == global_model_name:
            revision = global_model_revision

        # Enhanced epoch transition handling - WITH RANK SYNCHRONIZATION
        # Check if we're attempting to load a new epoch (not in cache)
        is_new_epoch_attempt = (
            hasattr(self, "local_progress") and 
            hasattr(self.local_progress, "epoch") and 
            self.local_progress.epoch is not None and 
            epoch != self.local_progress.epoch
        )
        
        # ONLY MASTER RANK does the checks to ensure all ranks make the same decision
        if self.master:
            # First do a quick check
            model_exists = check_model_exists(
                self,
                r2,
                local_model_name,
                prefix=prefix,
                revision=revision,
            )
            
            # If model doesn't exist and we're transitioning to a new epoch,
            # go DIRECTLY to global model fallback (no waiting)
            if not model_exists and is_new_epoch_attempt:
                self.logger.info(
                    f"🔄 Detected epoch transition: {self.local_progress.epoch} → {epoch}"
                )
                self.logger.info(
                    f"⚡ Local epoch-{epoch} model not available. Using global model directly..."
                )
                
                # Try loading from global model repository immediately (no retry delay)
                global_r2 = get_r2_client(self, uid=self.master_uid, donwload_on_all_ranks=True)
                global_prefix = f"epoch-{epoch}/"
                
                model_exists = check_model_exists(
                    self,
                    global_r2,
                    global_model_name,
                    prefix=global_prefix,
                    revision=global_model_revision,
                )
                
                if model_exists:
                    self.logger.info(
                        f"✅ Found model in global repository {global_model_name}"
                    )
                    # Update parameters to use global model
                    r2 = global_r2
                    local_model_name = global_model_name
                    prefix = global_prefix
                    revision = global_model_revision
                else:
                    # Last resort: try previous epoch from global model
                    self.logger.warning(
                        f"⚠️ Global epoch-{epoch} model also not available"
                    )
                    if epoch > 0:
                        prev_epoch = epoch - 1
                        self.logger.info(
                            f"🔄 Final fallback: trying previous epoch-{prev_epoch} from global model..."
                        )
                        prev_prefix = f"epoch-{prev_epoch}/"
                        prev_revision = f"{__run__}.{prev_epoch}.0"
                        
                        model_exists = check_model_exists(
                            self,
                            global_r2,
                            global_model_name,
                            prefix=prev_prefix,
                            revision=prev_revision,
                        )
                        
                        if model_exists:
                            self.logger.info(
                                f"✅ Using previous epoch-{prev_epoch} model as fallback"
                            )
                            r2 = global_r2
                            local_model_name = global_model_name
                            prefix = prev_prefix
                            revision = prev_revision
                            # Update epoch to the previous one since we're using that model
                            epoch = prev_epoch
            
            # Broadcast the fallback decision to all ranks
            # Use a tensor to communicate: [model_exists_flag, use_global_model, fallback_epoch]
            decision = torch.tensor(
                [
                    1 if model_exists else 0,
                    1 if local_model_name == global_model_name else 0,
                    epoch
                ], 
                dtype=torch.long, 
                device="cuda"
            )
        else:
            # Non-master ranks wait for the decision
            decision = torch.zeros(3, dtype=torch.long, device="cuda")
        
        # Broadcast master's decision to all ranks
        import torch.distributed as dist
        dist.broadcast(decision, src=0)
        
        # All ranks apply the same decision
        model_exists = bool(decision[0].item())
        use_global_model = bool(decision[1].item())
        epoch = int(decision[2].item())
        
        if not self.master:
            # Non-master ranks need to update their variables based on master's decision
            if use_global_model:
                global_r2 = get_r2_client(self, uid=self.master_uid, donwload_on_all_ranks=True)
                r2 = global_r2
                local_model_name = global_model_name
                prefix = f"epoch-{epoch}/"
                revision = f"{__run__}.{epoch}.0"
                self.logger.info(f"📡 Rank {self.local_rank}: Following master's decision to use global model epoch-{epoch}")
        
        if not model_exists:
            # Ensure inner_optimizer exists even if model loading fails
            if reload_inner_optimizer and not hasattr(self, "inner_optimizer") and hasattr(self, "model"):
                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate_maximum,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                )
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.inner_optimizer,
                    num_warmup_steps=1000,
                    num_training_steps=88000,
                )
                self.logger.info(f"Created Inner Optimizer before failure")
            raise Exception(f"Failed to load model. Check model exists failed.")

        if not hasattr(self, "model"):
            if use_cache is False:
                # Smart download: skip config if it already exists locally (architecture doesn't change)
                config_path = os.path.join(output_dir, "config.json")
                if os.path.exists(config_path):
                    self.logger.info("♻️  Reusing existing config.json (architecture unchanged)")
                else:
                    self.logger.info("📥 Downloading Config State")
                    _ = r2_download(
                        self,
                        r2=r2,
                        bucket=local_model_name,
                        key=f"{prefix}config.json",
                        donwload_on_all_ranks=False,
                        run_on_all_ranks=True,
                        destination=output_dir,
                    )
                self.logger.info("📥 Downloading Model Weights (safetensors)")
                _ = r2_download(
                    self,
                    r2=r2,
                    bucket=local_model_name,
                    key=f"{prefix}model.safetensors",
                    donwload_on_all_ranks=False,
                    run_on_all_ranks=True,
                    destination=output_dir,
                )
            self.logger.info("Setting Model")
            self.model = AutoModelForCausalLM.from_pretrained(
                output_dir,  # directory containing model files
                trust_remote_code=False,
            )
            self.logger.info("Loaded Model State")

            need_full_state = self.master and not hasattr(self, "outer_optimizer")
            if need_full_state:
                full_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,  # match your autocast compute dtype
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,  # required by FSDP2 policy
            )

            # Build a 1D device mesh over all ranks
            self.mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))

            # Keep a plain HF module and enable FSDP2 on it
            fully_shard(self.model, mesh=self.mesh, mp_policy=mp_policy)
            self.logger.info("Sharded Model State")
        else:
            # Model already loaded - only need to update weights
            if use_cache is False:
                self.logger.info("♻️  Model already loaded - downloading only updated weights")
                saftensors_path = r2_download(
                    self,
                    r2=r2,
                    bucket=local_model_name,
                    key=f"{prefix}model.safetensors",
                    donwload_on_all_ranks=False,
                    run_on_all_ranks=True,
                    destination=output_dir,
                )
            else:
                saftensors_path = os.path.join(self.output_dir, "model.safetensors")

            if self.master:
                model_state = load_file(saftensors_path, device="cpu")
            else:
                model_state = None

            self.logger.info(self.print_memory_usage())
            self.logger.info("Downloaded Model State")
            model_loading_options = StateDictOptions(
                full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
            )
            dist.barrier()
            self.logger.info("Model State Dict")
            set_model_state_dict(self.model, model_state, options=model_loading_options)
            self.logger.info("Sharded Model State")
            self.logger.info(self.print_memory_usage())

            need_full_state = self.master and not hasattr(self, "outer_optimizer")
            if need_full_state:
                # full_state = m_blob
                full_state = model_state
            # del objs, model_state, m_blob
            del model_state
            gc.collect()

        self.logger.info(
            f"Successfully Loaded Model From {local_model_name} With Revision {revision}"
        )

        # Move model to device
        self.model.config.block_list = []
        self.local_progress.inner_step = (
            self.model.config.inner_step
            if "inner_step" in self.model.config.__dict__
            else 0
        )
        if (local_model_name == global_model_name) and (
            epoch == self.global_progress.epoch
        ):
            self.allreduce_status_dict = (
                self.model.config.all_reduce_scores
                if "all_reduce_scores" in self.model.config.__dict__
                else {}
            )
        # if reload_inner_optimizer and not hasattr(self, "inner_optimizer"):
        if reload_inner_optimizer:
            if not hasattr(self, "inner_optimizer"):
                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate_maximum,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                )

                self.logger.info(f"Loaded Inner Optimizer")

                self.scheduler = get_cosine_schedule_with_warmup(
                    self.inner_optimizer,
                    num_warmup_steps=1000,
                    num_training_steps=88000,
                )
            # Load optimizer state with corruption detection and retry logic
            MAX_LOAD_RETRIES = 3
            load_attempt = 0
            optimizer_state = None
            optimizer_state_path = None
            
            while load_attempt < MAX_LOAD_RETRIES:
                try:
                    # Download or get path to optimizer state file
                    if use_cache is False:
                        optimizer_state_path = r2_download(
                            self,
                            r2=r2,
                            bucket=local_model_name,
                            key=f"{prefix}inner_optimizer.rank{self.local_rank+1:04d}-of-{self.world_size}.pt",
                            donwload_on_all_ranks=True,
                            run_on_all_ranks=True,
                            destination=output_dir,
                        )
                    else:
                        optimizer_state_path = os.path.join(
                            self.output_dir,
                            f"inner_optimizer.rank{self.local_rank+1:04d}-of-{self.world_size}.pt",
                        )
                    
                    # Validate file exists and has reasonable size
                    if not os.path.exists(optimizer_state_path):
                        raise FileNotFoundError(
                            f"Optimizer state file not found: {optimizer_state_path}"
                        )
                    
                    file_size = os.path.getsize(optimizer_state_path)
                    if file_size == 0:
                        raise ValueError(
                            f"Optimizer state file is empty (0 bytes): {optimizer_state_path}"
                        )
                    
                    if file_size < 1024:  # Less than 1KB is suspicious
                        self.logger.warning(
                            f"Optimizer state file is unusually small ({file_size} bytes): {optimizer_state_path}"
                        )
                    
                    # Attempt to load the checkpoint
                    self.logger.info(
                        f"Loading optimizer state from {optimizer_state_path} "
                        f"(attempt {load_attempt + 1}/{MAX_LOAD_RETRIES}, size: {file_size} bytes)"
                    )
                    optimizer_state = torch.load(
                        optimizer_state_path, map_location="cpu", weights_only=True
                    )
                    
                    # Validate loaded state structure
                    if not isinstance(optimizer_state, dict):
                        raise ValueError(
                            f"Loaded optimizer state is not a dictionary: {type(optimizer_state)}"
                        )
                    
                    if "optimizer_state_dict" not in optimizer_state:
                        raise ValueError(
                            "Loaded optimizer state missing required key 'optimizer_state_dict'"
                        )
                    
                    self.logger.info(f"Successfully loaded optimizer state (size: {file_size} bytes)")
                    break  # Success - exit retry loop
                    
                except (RuntimeError, ValueError, FileNotFoundError) as e:
                    error_msg = str(e)
                    is_corruption_error = (
                        "PytorchStreamReader" in error_msg or
                        "failed finding central directory" in error_msg or
                        "zip archive" in error_msg.lower() or
                        "corrupted" in error_msg.lower()
                    )
                    
                    load_attempt += 1
                    
                    if is_corruption_error or isinstance(e, (FileNotFoundError, ValueError)):
                        self.logger.warning(
                            f"❌ Failed to load optimizer state (attempt {load_attempt}/{MAX_LOAD_RETRIES}): {error_msg}"
                        )
                        
                        # Delete corrupted file if it exists
                        if optimizer_state_path and os.path.exists(optimizer_state_path):
                            try:
                                os.remove(optimizer_state_path)
                                self.logger.info(
                                    f"🗑️  Deleted corrupted file: {optimizer_state_path}"
                                )
                            except Exception as delete_error:
                                self.logger.warning(
                                    f"Failed to delete corrupted file: {delete_error}"
                                )
                        
                        # If we have retries left and not using cache, try re-downloading
                        if load_attempt < MAX_LOAD_RETRIES and use_cache is False:
                            self.logger.info(
                                f"🔄 Retrying download and load (attempt {load_attempt + 1}/{MAX_LOAD_RETRIES})..."
                            )
                            time.sleep(1)  # Brief delay before retry
                            continue
                        elif load_attempt < MAX_LOAD_RETRIES and use_cache is True:
                            # If using cache and file is corrupted, try downloading fresh copy
                            self.logger.info(
                                f"🔄 Cache file corrupted, attempting fresh download (attempt {load_attempt + 1}/{MAX_LOAD_RETRIES})..."
                            )
                            use_cache = False  # Switch to download mode
                            time.sleep(1)
                            continue
                    
                    # If it's not a corruption error or we've exhausted retries, re-raise
                    if load_attempt >= MAX_LOAD_RETRIES:
                        self.logger.error(
                            f"❌ Failed to load optimizer state after {MAX_LOAD_RETRIES} attempts. "
                            f"Last error: {error_msg}"
                        )
                        raise
                    else:
                        # Unexpected error, re-raise immediately
                        raise
            
            if optimizer_state is None:
                raise RuntimeError(
                    f"Failed to load optimizer state after {MAX_LOAD_RETRIES} attempts"
                )
            
            self.logger.info(f"Downloaded Optimizer State")

            inner_optimizer_loading_options = StateDictOptions(
                full_state_dict=False, cpu_offload=True
            )

            set_optimizer_state_dict(
                model=self.model,
                optimizers=self.inner_optimizer,
                optim_state_dict=optimizer_state["optimizer_state_dict"],
                options=inner_optimizer_loading_options,
            )
            self.logger.info(f"Set Inner Optimizer State Dict")

            if "scheduler_state" in optimizer_state:
                # Load saved scheduler state to continue from where we left off
                # Using FULL warmup for stable gradients (matches top miners like UID 203)
                self.scheduler.load_state_dict(optimizer_state["scheduler_state"])

            self.logger.info(
                f"Successfully Loaded Inner Optimizer State From {local_model_name} For Revision {revision}"
            )
            loading_success = 1

    except Exception as e:
        loading_success = 0
        self.logger.exception(
            "Failed to load inner optimizer state"
            f"(repo_id={local_model_name}, rev={revision}, rank={self.local_rank}, world={self.world_size})"
        )
        return loading_success

    finally:
        if optimizer_state is not None:
            if isinstance(optimizer_state, dict):
                keys = list(optimizer_state.keys())
                for k in keys:
                    del optimizer_state[k]
                    gc.collect()
            del optimizer_state
            gc.collect()
        torch.cuda.empty_cache()

    if self.master:
        if not hasattr(self, "outer_optimizer"):
            # Set outer optimizer
            optimizer = partial(torch.optim.SGD, lr=0.8, momentum=0.9, nesterov=True)

            # param_groups, main_parameters, parameter_names = check_params(optimizer, self.model.parameters(), None)
            param_groups, main_parameters, parameter_names = check_params(
                optimizer, full_state.values(), None
            )
            self.logger.info("Check Params")

            del full_state
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            self.status_loglevel = logging.DEBUG
            self.offload_optimizer = True
            self.custom_gradients = True
            self.reuse_tensors = True
            self.delta_rule_averaging = False
            self._old_tensors: Optional[Sequence[torch.Tensor]] = None  # for delta rule
            scheduler = None
            initialize_optimizer = True

            self.main_parameters, self.parameter_names = (
                main_parameters,
                parameter_names,
            )

            self.outer_optimizer, _ = init_components(
                self,
                main_parameters,
                param_groups,
                optimizer,
                scheduler,
                initialize_optimizer,
            )
            self.logger.info("Init Components")

            # Load a new gradient averager
            self.grad_averager = DTGradAverager(
                dht=self.dht,
                main_parameters=main_parameters,
                offloaded_optimizer=self.outer_optimizer,
                compression=hivemind.Quantile8BitQuantization(),
                prefix=f"{self.config.neuron.run_id}_grad_averager",
                min_group_size=self.config.neuron.min_group_size,
                min_matchmaking_time=30.0,
                request_timeout=10.0,
                next_chunk_timeout=45.0,
                allreduce_timeout=self.allreduce_timeout - 30.0 - 15.0,
                local_rank=self.local_rank,
                world_size=self.world_size,
                start=True,
            )
            self.logger.info("Successfully Loaded Gradient Averager")

        if reload_outer_optimizer:
            optimizer_state = None
            try:
                if use_global_cache is False:
                    optimizer_state_path = r2_download(
                        self,
                        r2=get_r2_client(
                            self, uid=self.master_uid, donwload_on_all_ranks=True
                        ),
                        bucket=global_model_name,
                        key=f"{prefix}outer_optimizer.pt",
                        donwload_on_all_ranks=False,
                        run_on_all_ranks=False,
                        destination=global_output_dir,
                    )
                else:
                    optimizer_state_path = os.path.join(
                        global_output_dir, "outer_optimizer.pt"
                    )
                optimizer_state = torch.load(
                    optimizer_state_path, map_location="cpu", weights_only=True
                )

                # Load optimizer state if available
                if "optimizer_state_dict" in optimizer_state:
                    self.outer_optimizer.load_state_dict(
                        optimizer_state["optimizer_state_dict"]
                    )

                self.logger.info(
                    f"Successfully Loaded Outer Optimizer State From {global_model_name} For Revision {'.'.join(revision.split('.')[:-1] + ['0'])}"
                )
                loading_success = 1

            except Exception as e:
                # TODO might need to remove these
                loading_success = 0
                self.logger.warning(
                    f"No optimizer state found or failed to load: {str(e)}. Initializing fresh optimizer."
                )

            finally:
                if isinstance(optimizer_state, dict):
                    keys = list(optimizer_state.keys())
                    for k in keys:
                        del optimizer_state[k]
                        gc.collect()
                del optimizer_state
                gc.collect()
                torch.cuda.empty_cache()

        self.avg_handler = AveragingHandler(
            self.model,
            self.inner_optimizer,
            self.outer_optimizer,
            self.grad_averager,
            self.retry_limit,
            self.retry_delay,
            self.uid,
            self.config.neuron.local_batch_size_train,
            self.config.neuron.local_batch_size_train_effective,
            self.tokenizer,
            self.device,
            self.logger,
            # parameters_list,
        )
        if (
            (self.master)
            and (self.local_progress.inner_step != 0)
            and ("." in revision)
        ):
            self.avg_handler.reset_main_parameters(
                r2, global_model_name, prefix, use_global_cache, global_output_dir
            )

    self.logger.info(f"After Loading State {self.print_memory_usage()}")
    return loading_success


def load_state_from_peer(
    self,
    uid=None,
    epoch=None,
    reload_inner_optimizer=True,
    reload_outer_optimizer=True,
    revision=None,
    use_fallback_model=True,
):
    try:
        state_loaded = False
        if epoch is None:
            self.global_progress.epoch = get_progress(self, "global")[0]
            epoch = self.global_progress.epoch
        if uid is None:
            uid = self.master_uid
        if uid == self.master_uid:
            self.local_progress.inner_step = get_progress(
                self, "global", self.config.global_model_name, epoch=epoch
            )[1]
        else:
            self.local_progress.inner_step = get_progress(
                self, "local", self.config.r2.bucket_name, epoch=epoch
            )[1]

        # self.logger.debug("Model Weights Before Loading State")
        # current_model_weights_sample = copy.copy(
        #     [layer for layer in self.model.parameters()][-2][-10:].tolist()
        # )
        # self.logger.debug(current_model_weights_sample)

        self.logger.debug(f"Old Model Tag: {self.local_progress.epoch}")
        
        # Log epoch transition detection
        if (
            hasattr(self, "local_progress") and 
            hasattr(self.local_progress, "epoch") and 
            self.local_progress.epoch is not None and 
            epoch != self.local_progress.epoch
        ):
            self.logger.info(
                f"🔄 EPOCH TRANSITION DETECTED: Current epoch {self.local_progress.epoch} → Target epoch {epoch}"
            )
            self.logger.info(
                f"📡 This may require waiting for validators to upload new epoch models to R2..."
            )

        if self.global_progress.epoch is not None:
            self.logger.debug(
                f"Latest Model State Found On The HF Hub With The Tag: {self.global_progress.epoch}. Loading That Model State."
            )

            # Load model state with max retries
            MAX_ATTEMPTS = 3
            attempt = 0

            while attempt < MAX_ATTEMPTS:
                try:
                    loading_success = torch.tensor(
                        [
                            load_model_optimizer_gradient_averager(
                                self,
                                uid=uid,
                                epoch=epoch,
                                reload_inner_optimizer=reload_inner_optimizer,
                                reload_outer_optimizer=reload_outer_optimizer,
                                revision=revision,
                            )
                        ]
                    )
                    dist.all_reduce(loading_success, group=self.gloo_group)
                    if (loading_success[0].item() != self.world_size) and (
                        uid != self.master_uid
                    ):
                        self.logger.info(
                            "Failed to load local model. Loading global model"
                        )
                        self.logger.info(
                            "LOAD GLOBAL MODEL",
                            load_model_optimizer_gradient_averager(
                                self,
                                uid=self.master_uid,
                                epoch=self.global_progress.epoch,
                            ),
                        )
                    break

                except Exception as e:
                    attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        if use_fallback_model:
                            # TODO Crash the whole process if global model is not loaded
                            loading_success = torch.tensor(
                                [
                                    load_model_optimizer_gradient_averager(
                                        self,
                                        uid=self.master_uid,
                                        epoch=self.global_progress.epoch,
                                    ),
                                ]
                            )
                        else:
                            raise Exception(
                                f"Failed to load model after {MAX_ATTEMPTS} attempts: {str(e)}"
                            )
                    self.logger.info(
                        f"Failed to load model, retrying. Attempt {attempt}/{MAX_ATTEMPTS}. Error {str(e)}"
                    )
                    self.logger.info(e)

            state_loaded = True

            # self.logger.debug("Model Weights After Loading State")
            # new_model_weights_sample = copy.copy(
            #     [layer for layer in self.model.parameters()][-2][-10:].tolist()
            # )
            # self.logger.debug(new_model_weights_sample)

            self.local_progress.epoch = epoch
            self.local_progress.samples_accumulated = 0
            self.logger.info(f"New Model Tag: {self.global_progress.epoch}")

            cleanup_old_cache(self)

        else:
            self.logger.debug(f"Model With Tag: {epoch} Does Not Exist")

        # Ensure inner_optimizer is always created after loading state
        if hasattr(self, "model") and not hasattr(self, "inner_optimizer"):
            self.logger.warning("Inner optimizer not created during state load. Creating now...")
            self.inner_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate_maximum,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
            self.scheduler = get_cosine_schedule_with_warmup(
                self.inner_optimizer,
                num_warmup_steps=1000,
                num_training_steps=88000,
            )
            # Using FULL warmup for stable gradients (matches top miners like UID 203)
            self.logger.info(f"Created Inner Optimizer as fallback")
        
        return state_loaded

    except Exception as e:
        self.logger.error(f"Error loading state: {str(e)}")
        
        # Ensure inner_optimizer exists even on exception, if model exists
        if hasattr(self, "model") and not hasattr(self, "inner_optimizer"):
            self.logger.warning("Creating inner optimizer after exception during state load...")
            try:
                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate_maximum,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                )
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.inner_optimizer,
                    num_warmup_steps=1000,
                    num_training_steps=88000,
                )
                # Using FULL warmup for stable gradients (matches top miners like UID 203)
                self.logger.info(f"Created Inner Optimizer as exception handler fallback")
            except Exception as opt_error:
                self.logger.error(f"Failed to create inner_optimizer: {opt_error}")
        
        return False


def cleanup_old_cache(self):
    """Helper method to clean up old cache files"""
    if self.master:
        files = [
            file
            for file in os.listdir(os.getcwd())
            if os.path.isdir(file)
            and (self.config.neuron.global_model_name in file)
            and (file != self.config.r2.bucket_name)
            and (file != self.config.neuron.global_model_name)
        ]
        for file in files:
            try:
                shutil.rmtree(file, ignore_errors=True)
                self.logger.info(f"Deleting cache for model {file}")
            except Exception as e:
                self.logger.info(f"Error deleting cache for model {file}: {e}")
    return


def upload_new_state(
    self,
    epoch: int,
    results: dict,
    model_state: dict,
    inner_optimizer_state: dict,
    inner_optimizer_lr: int,
    block: int = None,
):
    # Save and upload both model and optimizer state
    upload_success = save_and_upload_state(
        self,
        epoch=epoch,
        results=results,
        model_state=model_state,
        inner_optimizer_state=inner_optimizer_state,
        inner_optimizer_lr=inner_optimizer_lr,
        block=block,
    )
    upload_success_status = torch.tensor([1]) if upload_success else torch.tensor([0])
    dist.all_reduce(upload_success_status, group=self.gloo_group)
    if (upload_success_status[0].item() == self.world_size) and self.master:
        self.logger.info(
            f"Successfully pushed new model with tag {self.local_progress.epoch}"
        )
    elif upload_success_status[0].item() != self.world_size:
        self.logger.error(
            "Maximum Retry Limit Reached. Unable To Upload Model To HF Hub."
        )

    return upload_success


def save_and_upload_state(
    self,
    epoch: int,
    results: dict,
    model_state: dict,
    inner_optimizer_state: dict,
    inner_optimizer_lr: int,
    block: int = None,
):
    """Unified function to save and upload both model and optimizer state"""
    if self.master:
        batch_size = sum(
            [result for result in results["gathered"].values() if result is not None]
        )
        participating_peers = results["participating_peers"]
        failed_peers = results["failed_peers"]

    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            if self.master:
                self.logger.info(
                    f"Preparing model and optimizer state for epoch {epoch}"
                )
                if block is not None:
                    self.model.config.last_allreduce_block = block
                self.model.config.inner_step = 0
                self.model.config.save_pretrained(self.output_dir)
                self.logger.info("Save config")
                save_file(
                    model_state,
                    os.path.join(self.output_dir, "model.safetensors"),
                    metadata={"format": "pt"},
                )
                self.logger.info("Save model")

                # Save outer optimizer state
                outer_optimizer_state = {
                    "optimizer_state_dict": self.outer_optimizer.state_dict(),
                    "learning_rate": self.outer_optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
                torch.save(
                    outer_optimizer_state,
                    os.path.join(self.output_dir, "outer_optimizer.pt"),
                )
                self.logger.info("Save optimizer")

            # Save inner optimizer state
            inner_optimizer_state = {
                "optimizer_state_dict": inner_optimizer_state,
                "learning_rate": inner_optimizer_lr,
                "scheduler_state": self.scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                inner_optimizer_state,
                os.path.join(
                    self.output_dir,
                    f"inner_optimizer.rank{self.local_rank+1:04d}-of-{self.world_size}.pt",
                ),
            )

            if self.master:
                self.logger.info(
                    f"Uploading model and optimizer states to bucket: {self.config.r2.bucket_name}"
                )
                # Upload everything in one go
                upload_folder_to_r2(
                    r2=self.r2["write"],
                    bucket=self.config.r2.bucket_name,
                    prefix=f"epoch-{epoch}/",
                )

                log_peerid_to_r2(self, prefix=f"epoch-{self.local_progress.epoch}/")
                log_peerid_to_r2(self)

            else:
                time.sleep(1000)
            self.logger.info(
                f"Successfully pushed new model and optimizer state with tag {epoch} to bucket: {self.config.r2.bucket_name}"
            )
            return True

        except Exception as e:
            attempt += 1
            self.logger.warning(
                f"Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                self.logger.error(
                    "Maximum retry limit reached. Unable to upload state to HF Hub."
                )
                # raise
                return False
    return False
