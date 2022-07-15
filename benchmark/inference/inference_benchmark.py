import argparse
import copy
from timeit import default_timer

import torch
from utils import get_dataset, get_model

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'reddit': ['edge_conv', 'gat', 'gcn', 'pna_conv'],
    'ogbn-products': ['edge_conv', 'gat', 'gcn', 'pna_conv'],
}


def run(args: argparse.ArgumentParser) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('BENCHMARK STARTS')
    if args.pure_gnn_mode:
        print('PURE GNN MODE ACTIVATED')
    for dataset_name in args.datasets:
        print(f'Dataset: {dataset_name}')
        dataset = get_dataset(dataset_name, args.root)

        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None
        degree = None

        data = dataset[0].to(device)
        inputs_channels = data.x_dict['paper'].size(
            -1) if dataset_name == 'ogbn-mag' else dataset.num_features

        for model_name in args.models:
            if model_name not in supported_sets[dataset_name]:
                print(f'Configuration of {dataset_name} + {model_name} '
                      f'not supported. Skipping.')
                continue
            print(f'Evaluation bench for {model_name}:')

            for batch_size in args.eval_batch_sizes:
                if dataset_name != 'ogbn-mag':
                    subgraph_loader = NeighborLoader(
                        copy.copy(data),
                        num_neighbors=[-1],
                        input_nodes=mask,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                    )
                    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

                for layers in args.num_layers:
                    if dataset_name == 'ogbn-mag':
                        subgraph_loader = NeighborLoader(
                            copy.copy(data),
                            num_neighbors=[-1] * layers,
                            input_nodes=mask,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                        )
                        subgraph_loader.data.n_id = torch.arange(
                            data.num_nodes)

                    for hidden_channels in args.num_hidden_channels:
                        print(
                            '-----------------------------------------------')
                        print(f'Batch size={batch_size}, '
                              f'Layers amount={layers}, '
                              f'Hidden features size={hidden_channels}')
                        params = {
                            'inputs_channels': inputs_channels,
                            'hidden_channels': hidden_channels,
                            'output_channels': dataset.num_classes,
                            'num_heads': args.num_heads,
                            'num_layers': layers,
                        }

                        if model_name == 'pna_conv':
                            if degree is None:
                                degree = PNAConv.get_degree(subgraph_loader)
                                print(f'Calculated degree for {dataset_name}.')
                            params['degree'] = degree

                        model = get_model(
                            model_name, params, metadata=data.metadata()
                            if dataset_name == 'ogbn-mag' else None)
                        model = model.to(device)
                        model.training = False
                        progress_bar = True
                        if args.pure_gnn_mode:
                            prebatched_samples = []
                            for i, batch in enumerate(subgraph_loader):
                                if i == args.prebatched_samples:
                                    break
                                prebatched_samples.append(batch)
                            subgraph_loader = prebatched_samples

                        start = default_timer()
                        model.inference(subgraph_loader, device, progress_bar)
                        stop = default_timer()
                        print(f'Inference time={stop-start:.3f}\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')
    argparser.add_argument(
        '--pure-gnn-mode', action='store_true',
        help='turn on pure gnn efficiency bench - firstly prepare batches')
    argparser.add_argument('--prebatched_samples', default=3, type=int,
                           help='number of preloaded batches in pure_gnn mode')
    argparser.add_argument('--datasets', nargs='+',
                           default=['ogbn-mag', 'ogbn-products',
                                    'reddit'], type=str)
    argparser.add_argument(
        '--models', nargs='+',
        default=['edge_conv', 'gat', 'gcn', 'pna_conv', 'rgat',
                 'rgcn'], type=str)
    argparser.add_argument('--root', default='../../data', type=str)
    argparser.add_argument('--eval-batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', nargs='+',
                           default=[64, 128, 256], type=int)
    argparser.add_argument(
        '--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    argparser.add_argument('--num-workers', default=2, type=int)

    args = argparser.parse_args()

    run(args)
