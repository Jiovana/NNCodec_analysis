'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

### model determinism / reproducibility
import os
SEED_RANDOM = 909
SEED_TORCH = 808
SEED_NUMPY = 303
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Must be before torch import
os.environ["PYTHONHASHSEED"] = str(SEED_RANDOM)
os.environ["RAY_RANDOM_SEED"] = str(SEED_RANDOM)
os.environ["RAY_DEDUP_LOGS"] = "0"

import torch
import random
import numpy as np

# PyTorch seeding
torch.manual_seed(SEED_TORCH)
torch.cuda.manual_seed_all(SEED_TORCH)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    torch.mps.manual_seed(SEED_TORCH)
    torch.mps.set_per_process_memory_fraction(0.75)

random.seed(SEED_RANDOM)
np.random.seed(SEED_NUMPY)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
from nncodec.tensor import encode, decode

parser = argparse.ArgumentParser(description='NNCodec for Tensors in AI-based Media Processing (TAIMP)')
parser.add_argument('--qp', type=int, default=-32, help='quantization parameter (default: -32)')
parser.add_argument('--nonweight_qp', type=int, default=-75, help='qp for non-weights, e.g. BatchNorm params (default: -75)')
parser.add_argument('--bitdepth', type=int, default=8, help='Optional: integer-aligned bitdepth for limited precision (default: None); note: overwrites QPs.')
parser.add_argument('--approx_method', type=str, default='uniform',  help='Approximation method [uniform or codebook]')
parser.add_argument('--opt_qp', action='store_true', help='Modifies QP layer-wise')
parser.add_argument('--use_dq', action='store_true', help='Enable dependent scalar / Trellis-coded quantization')
parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity rate (default: 0.0)')
parser.add_argument('--struct_spars_factor', type=float, default=0.0, help='Factor for structured sparsification (default: 0.0, recommended 0.75-0.95)')
parser.add_argument('--row_skipping', action='store_true', help='Enable Row Skipping')
parser.add_argument('--tca', action='store_true', help='Enable Temporal Context Adaptation')
parser.add_argument('--tensor_path', type=str, default=None, metavar='e.g., ./tensor.pt or ./tensor.npy')
parser.add_argument('--job_identifier', type=str, default='TensorAIMP', metavar='TensorAIMP')
parser.add_argument('--tensor_id', type=str, default='0', metavar='identifier for coded tensor in the bitstream (default: "0")')
parser.add_argument('--results', type=str, default='.')
parser.add_argument('--verbose', action='store_true', help='Stdout process information.')
parser.add_argument('--compress_differences', action='store_true', help='Compresses weight differences wrt. to base model, otherwise full base models are communicated')
parser.add_argument('--err_accumulation', action='store_true', help='Locally accumulates quantization errors (residuals)')
parser.add_argument('--cuda_device', type=int, default=None)
parser.add_argument('--quantize_only', action='store_true', help='return quantized tensor instead of entropy-coded bitstream')
parser.add_argument('--incremental', action='store_true', help='starts incremental test pipeline, e.g., for Temporal Context Adaptation')

def main():
    args = parser.parse_args()
    args_dict = vars(args)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if torch.cuda.is_available():
            if args.cuda_device is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if args.tensor_path is not None:
        example_tensor = None
    else:
        example_tensor = torch.randn(256, 64, 64, device=device)
        # example_tensor = torch.randint(0, 255, (3, 3, 32, 32), device=device) # example integer tensor
        # example_tensor = np.random.rand(256, 64, 64) # example numpy tensor

    if not args.incremental:
        # --- NNCodec ---
        bitstream = encode(example_tensor, args_dict)
        dec_tensor = torch.tensor(decode(bitstream, args_dict["tensor_id"]), device=device, dtype=torch.float32)
        # ---------------
    else:
        approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}} if args.tca else None
        num_increments = 5
        delta = 0.001 # amount of incremental update to masked tensor elements
        mask = (torch.rand_like(example_tensor) < 0.5)
        sign_mask = mask * (2 * torch.randint(0, 2, example_tensor.shape, device=device, dtype=torch.float32) - 1)
        for _ in range(num_increments):
            # --- NNCodec ---
            bitstream = encode(example_tensor, args_dict, approx_param_base)
            dec_tensor = torch.tensor(decode(bitstream, args_dict["tensor_id"], approx_param_base), device=device, dtype=torch.float32)
            # ---------------
            example_tensor = example_tensor + (delta * sign_mask)

if __name__ == '__main__':
    main()
