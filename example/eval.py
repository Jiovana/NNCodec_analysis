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

# Global seed constants
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
torch.use_deterministic_algorithms(True, warn_only=False)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    torch.mps.manual_seed(SEED_TORCH)
    torch.mps.set_per_process_memory_fraction(0.75)

random.seed(SEED_RANDOM)
np.random.seed(SEED_NUMPY)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import copy
def seeded_model_copy(model, seed=SEED_TORCH):
    torch.manual_seed(seed)
    return copy.deepcopy(model)


import argparse
import shutil
import torchvision
import wandb
import warnings
from nncodec import nnc
from nncodec import nnc_core
from nncodec.framework import pytorch_model
from nncodec.framework.applications.utils.sparsification import apply_struct_spars_v2, apply_unstruct_spars_v2, get_sparsity
from nncodec.framework.applications.utils.transforms import split_datasets
from nncodec.framework.applications import models, datasets


parser = argparse.ArgumentParser(description='Evaluation Script for TinyLlama on V2X')
parser.add_argument('--qp', type=int, default=-32, help='quantization parameter for NNs (default: -32)')
parser.add_argument('--qp_density', type=int, default=2, help='quantization scale parameter (default: 2)')
parser.add_argument('--nonweight_qp', type=int, default=-75, help='qp for non-weights, e.g. BatchNorm params (default: -75)')
parser.add_argument("--opt_qp", action="store_true", help='Modifies QP layer-wise')
parser.add_argument("--use_dq", action="store_true", help='Enable dependent scalar / Trellis-coded quantization')
parser.add_argument("--lsa", action="store_true", help='Enable Local Scaling Adaptation')
parser.add_argument("--bnf", action="store_true", help='Enable BatchNorm Folding')
parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity rate (default: 0.0)')
parser.add_argument('--struct_spars_factor', type=float, default=0.0, help='Factor for structured sparsification (default: 0.9)')
parser.add_argument("--row_skipping", action="store_true", help='Enable Row Skipping')
parser.add_argument("--tca", action="store_true", help='Enable Temporal Context Adaptation')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default=64)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--max_batches', type=int, default=None, help='Max num of batches to process (default: 0, i.e., all)')
parser.add_argument('--max_batches_test', type=int, default=None, help='Max num of batches to process in test fct (default: None, i.e., all)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
parser.add_argument('--model', type=str, default='resnet56', metavar=f'any of {models.__all__} or {torchvision.models.list_models(torchvision.models)}')
parser.add_argument('--model_path', type=str, default=None, metavar='./example/ResNet56_CIF100.pt')
parser.add_argument('--model_rand_int', action="store_true", help='model randomly initialized, i.e., w/o loading pre-trained weights')
parser.add_argument('--dataset', type=str, default='cifar100', metavar=f"Any of {datasets.__all__}")
parser.add_argument('--dataset_path', type=str, default='../data')
parser.add_argument('--tokenizer_path', type=str, default='./tokenizer/telko_tokenizer.model')
parser.add_argument('--max_seq_len', type=int, default=1525, help='Custom max_seq_len for tiny Llama')
parser.add_argument('--TLM_size', type=int, default=0, help='tiny Llama size [0, 1, 2, 3]')
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers (default: 4), if 0 debugging mode enabled')
parser.add_argument("--wandb", action="store_true", help='Use Weights & Biases for data logging')
parser.add_argument('--wandb_key', type=str, default='', help='Authentication key for Weights & Biases API account ')
parser.add_argument('--wandb_run_name', type=str, default='NNC_WP_spars', help='Identifier for current run')
parser.add_argument('--job_id', type=str, default='', help='Identifier for job e.g. on GPU Cluster')
parser.add_argument("--pre_train_model", action="store_true", help='Training the full model prior to compression')
parser.add_argument("--verbose", action="store_true", help='Stdout process information.')
parser.add_argument('--num_clients', type=int, default=2, help='Number of clients in FL scenario (default: 2)')
parser.add_argument("--compress_upstream", action="store_true", help='Compression of clients-to-server communication')
parser.add_argument("--compress_downstream", action="store_true", help='Compression of server-to-clients communication')
parser.add_argument("--compress_differences", action="store_true", help='Compresses weight differences wrt. to base model, otherwise full base models are communicated')
parser.add_argument("--err_accumulation", action="store_true", help='Locally accumulates quantization errors (residuals)')
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--cuda_device', type=int, default=None)

def main():
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    clear_results = False

    if torch.cuda.is_available() and args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if clear_results and os.path.exists(os.path.join(args.results)):#, f'd{args.model}_epoch0_qp_{args.qp}_bitstream.nnc')):
        shutil.rmtree(f"{args.results}")

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if args.wandb:
        if isinstance(args.wandb_key, str) and len(args.wandb_key) == 40:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        else:
            assert 0, "incompatible W&B authentication key"

    if args.model in models.__all__:
        if "tinyllama" in args.model:
            model, tokenizer = models.init_model(args.model, parser_args=args)
        else:
            model = models.init_model(args.model, num_classes=100)
    elif args.model in torchvision.models.list_models(torchvision.models):
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    elif args.model in torchvision.models.segmentation.deeplabv3.__all__:
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    else:
        assert 0, f"Model not specified in /framework/applications/models and not available in torchvision model zoo" \
                  f"{torchvision.models.list_models(torchvision.models)})"

    if not args.model_rand_int and args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    elif not args.model_rand_int and args.model_path and not os.path.exists(args.model_path):
        assert 0, f"No model found in {args.model_path}"

    state_dict = model.state_dict()

    # Prepare data for plotting
    param_names = []
    param_sizes = []

    for name, param in state_dict.items():
        print(f'{name}: shape {param.shape}, #{param.numel()}')
        param_names.append(name)
        param_sizes.append(param.numel())  # Number of elements in the parameter tensor
    total_params = sum(param_sizes)
    print(f"Total number of parameters: {total_params}")


    UCS = {'cifar100': 'NNR_PYT_CIF100', 'cifar10': 'NNR_PYT_CIF10', 'imagenet200': 'NNR_PYT_IN200',
           'voc': 'NNR_PYT_VOC', 'V2X': 'NNR_PYT_Telko'}
    use_case_name = UCS[args.dataset] if args.dataset in UCS else 'NNR_PYT'

    nnc_mdl, nnc_mdl_executer, mdl_params = pytorch_model.create_NNC_model_instance_from_object(
        model,
        dataset_path=args.dataset_path if not args.dataset in ['V2X'] else None,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lsa=args.lsa,
        epochs=args.epochs,
        use_case=use_case_name
    )

    mdl_info = {"parameter_index": {k: i for i, k in enumerate(nnc_mdl_executer.model.state_dict().keys())
                                            if nnc_mdl_executer.model.state_dict()[k].shape != torch.Size([])}}

    if args.bnf:
        p_types = nnc_mdl.guess_block_id_and_param_type(nnc_mdl_executer.model.state_dict(),
                                                        bn_info=nnc_mdl.get_torch_bn_info(nnc_mdl_executer.model))
        mdl_info = nnc.compress(mdl_params, model=nnc_mdl, bnf_mapping=True, block_id_and_param_type=p_types)

    # split datasets across clients
    if not args.dataset in ['V2X']:
        trainloaders, valloaders, testloader = split_datasets(nnc_mdl_executer.test_set,
                                                              nnc_mdl_executer.train_loader.dataset,
                                                              num_partitions=args.num_clients,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.workers,
                                                              val_ratio=0.0, #if > 0 creates validation split at each client
                                                              )
    else:
        testloader = datasets.V2X(args, test_only=True, shuffle=False)

    perf = nnc_mdl_executer.handle.evaluate(nnc_mdl_executer.model, criterion=nnc_mdl_executer.handle.criterion,
                                            testloader=testloader, device=nnc_mdl_executer.device, verbose=True,
                                            max_batches=args.max_batches_test, detokenize=True, args=args)
    print(f"Global test performance: {[str(k) + ': ' + str(v) for k, v in perf.items()]}")




if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    main()
