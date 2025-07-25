# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click


def common_args(func):
    """Decorator for common model arguments"""
    func = click.option("--model", type=str, help="Model name (e.g., 'llama3.1-8b')")(
        func
    )
    func = click.option(
        "--model_config", type=str, help="Path to a single model config YAML file"
    )(func)
    func = click.option(
        "--model_configs",
        type=str,
        help="Path to multiple model config YAML files (supports glob patterns like 'configs/*.yaml')",
    )(func)
    return func


def cli_args(func):
    """Decorator for CLI-specific arguments"""
    func = click.option("--pp", type=int, help="Pipeline parallelism size")(func)
    func = click.option("--tp", type=int, help="Tensor parallelism size")(func)
    func = click.option("--isl", type=int, help="Input sequence length")(func)
    func = click.option("--osl", type=int, help="Output sequence length")(func)
    func = click.option("--num_requests", type=int, help="Number of requests")(func)
    func = click.option("--page_size", type=int, help="Page size")(func)
    func = click.option("--access_pattern", type=str, help="Access pattern")(func)
    return func


def plan_args(func):
    """Decorator for plan-specific arguments"""
    func = click.option(
        "--format",
        default="text",
        type=str,
        help="Output of the nixl command [text, json, csv] (default: text)",
    )(func)
    return func


def nixl_bench_args(func):
    """Decorator for NIXL benchmark arguments"""
    func = click.option(
        "--source",
        default="file",
        type=str,
        help="Source of the nixl descriptors [file, memory, gpu] (default: file)",
    )(func)
    func = click.option(
        "--destination",
        default="memory",
        type=str,
        help="Destination of the nixl descriptors [file, memory, gpu] (default: memory)",
    )(func)
    func = click.option(
        "--backend", type=str, help="Communication backend [UCX, UCX_MO] (default: UCX)"
    )(func)
    func = click.option(
        "--worker_type",
        type=str,
        help="Worker to use to transfer data [nixl, nvshmem] (default: nixl)",
    )(func)
    func = click.option(
        "--initiator_seg_type",
        type=str,
        help="Memory segment type for initiator [DRAM, VRAM] (default: DRAM)",
    )(func)
    func = click.option(
        "--target_seg_type",
        type=str,
        help="Memory segment type for target [DRAM, VRAM] (default: DRAM)",
    )(func)
    func = click.option(
        "--scheme",
        type=str,
        help="Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise)",
    )(func)
    func = click.option(
        "--mode",
        type=str,
        help="Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG)",
    )(func)
    func = click.option(
        "--op_type", type=str, help="Operation type [READ, WRITE] (default: WRITE)"
    )(func)
    func = click.option(
        "--check_consistency", is_flag=True, help="Enable consistency checking"
    )(func)
    func = click.option(
        "--total_buffer_size", type=int, help="Total buffer size (default: 8GiB)"
    )(func)
    func = click.option(
        "--start_block_size", type=int, help="Starting block size (default: 4KiB)"
    )(func)
    func = click.option(
        "--max_block_size", type=int, help="Maximum block size (default: 64MiB)"
    )(func)
    func = click.option(
        "--start_batch_size", type=int, help="Starting batch size (default: 1)"
    )(func)
    func = click.option(
        "--max_batch_size", type=int, help="Maximum batch size (default: 1)"
    )(func)
    func = click.option(
        "--num_iter", type=int, help="Number of iterations (default: 1000)"
    )(func)
    func = click.option(
        "--warmup_iter", type=int, help="Number of warmup iterations (default: 100)"
    )(func)
    func = click.option(
        "--num_threads",
        type=int,
        help="Number of threads used by benchmark (default: 1)",
    )(func)
    func = click.option(
        "--num_initiator_dev",
        type=int,
        help="Number of devices in initiator processes (default: 1)",
    )(func)
    func = click.option(
        "--num_target_dev",
        type=int,
        help="Number of devices in target processes (default: 1)",
    )(func)
    func = click.option("--enable_pt", is_flag=True, help="Enable progress thread")(
        func
    )
    func = click.option(
        "--device_list", type=str, help="Comma-separated device names (default: all)"
    )(func)
    func = click.option(
        "--runtime_type", type=str, help="Type of runtime to use [ETCD] (default: ETCD)"
    )(func)
    func = click.option(
        "--etcd-endpoints",
        type=str,
        help="ETCD server URL for coordination (default: http://localhost:2379)",
    )(func)
    func = click.option(
        "--storage_enable_direct",
        type=bool,
        help="Enable direct I/O for storage operations (only used with POSIX backend)",
        default=False,
    )(func)
    func = click.option(
        "--gds_filepath", type=str, help="File path for GDS operations"
    )(func)
    func = click.option(
        "--enable-vmm",
        type=bool,
        help="Enable VMM memory allocation when DRAM is requested",
    )(func)
    return func


def ctp_args(func):
    """Decorator for CTP (Custom Traffic Pattern) arguments"""
    func = click.option(
        "--verify-buffers/--no-verify-buffers",
        default=False,
        help="Verify buffer contents after transfer",
    )(func)
    func = click.option(
        "--print-recv-buffers/--no-print-recv-buffers",
        default=False,
        help="Print received buffer contents",
    )(func)
    func = click.option(
        "--json-output-path",
        type=click.Path(),
        help="Path to save JSON output",
        default=None,
    )(func)
    func = click.option(
        "--config-file",
        type=click.Path(exists=True),
        required=True,
        help="Path to YAML config file",
    )(func)
    return func
