
import torch_geometric.transforms as T
import os.path as osp
import argparse
from timeit import default_timer
from torch_geometric.nn import to_hetero
from torch_geometric.loader import NeighborLoader
from utils import get_dataset, get_model
import copy
from torch_sparse import SparseTensor


supported_sets = {
    'ogbn-mag': ['rgcn','rgat'],
    'reddit': ['gcn'], #TODO ...
    'ogbn-products': ['gat', ],  #TODO ...
}

def run(args: argparse.ArgumentParser) -> None:

    print("BENCHMARK STARTS")
    if args.pure_gnn_mode:
        print("PURE GNN MODE ACTIVATED")
    for dataset_name in args.datasets:
        print("Dataset: ", dataset_name)
        dataset = get_dataset(dataset_name, args.root)

        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None

        data = dataset[0].to(args.device)
        inputs_channels =  data.x_dict['paper'].size(-1) if dataset_name == 'ogbn-mag' else dataset.num_features

        for model_name in args.models:
            if model_name not in supported_sets[dataset_name]:
                 print(f'Configuration of {dataset_name} + {model_name} not supported. Skipping.')
                 continue
            print(f'Evaluation bench for {model_name}:')
            for layers in args.num_layers:
                print(f"Layers amount={layers}")
                for hidden_channels in args.num_hidden_channels:
                    print(f"Hidden features size={hidden_channels}")
                    inputs_channels
                    params = {
                        'inputs_channels': inputs_channels,
                        'hidden_channels': hidden_channels,
                        'output_channels': dataset.num_classes,
                        'num_relations': args.num_relations,
                        'num_layers': layers,
                    }
                    # if dataset_name == 'ogbn-mag':
                    #     hetero_params = {
                    #         'num_nodes_dict': data.num_nodes_dict,
                    #         'x_types': list(data.x_dict.keys()),
                    #         'edge_types': list(data.adj_t_dict.keys())
                    #     }
                        #params = dict(params, **hetero_params)
                    #print(data.metadata())
                    model = get_model(model_name,
                                      params,
                                      args.device,
                                      metadata=data.metadata() if dataset_name == 'ogbn-mag' else None)

                    for batch_size in args.eval_batch_sizes:
                        subgraph_loader = NeighborLoader(copy.copy(data),
                                                        num_neighbors=[-1],
                                                        input_nodes=mask,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        )

                        if args.pure_gnn_mode:
                            i = 0
                            prebatched_samples = []
                            for batch in subgraph_loader:
                         #       print(batch)
                                if i == args.prebatched_samples:
                                    break
                                prebatched_samples.append(batch)
                                i += 1
                            subgraph_loader = prebatched_samples

                        start = default_timer()
                        model.inference(subgraph_loader,
                                        args.device)
                        stop =  default_timer()
                        print(f'Batch size={batch_size} Inference time={stop-start}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')

    argparser.add_argument('--device', default='cpu', type=str)
    argparser.add_argument('--pure-gnn-mode', default='False', choices=('True','False'), type=str,
                           help="turn on pure gnn efficiency bench - firstly prepare batches")
    argparser.add_argument('--prebatched_samples', default=5 , type=int,
                           help="number of preloaded batches in pure_gnn mode"),
    argparser.add_argument('--datasets', nargs="+",
                           default=['ogbn-mag'], type=str)
    argparser.add_argument('--models', nargs="+",
                           default=['rgcn', 'rgat'], type=str)
    argparser.add_argument('--root', default='../../data', type=str)
    argparser.add_argument('--eval-batch-sizes',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', default= [1, 2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', default= [64, 128, 256], type=int)
    argparser.add_argument('--num-relations', default=3, type=int)
    argparser.add_argument('--num-workers', default=0, type=int)

    args = argparser.parse_args()

    run(args)