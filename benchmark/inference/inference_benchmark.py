
import torch_geometric.transforms as T
import os.path as osp
import argparse
from timeit import default_timer
from torch_geometric.loader import NeighborLoader
from .utils import get_dataset, get_model
import copy


supported_sets = {
    'ogbn-mag': ['rgcn','rgat'],
    'reddit': ['gcn'], #TODO ...
}

def run(args: argparse.ArgumentParser) -> None:

    print("BENCHMARK STARTS")
    for dataset in args.datasets:
        print("Dataset: ", dataset)
        dataset = get_dataset(dataset, args.root)

        mask = ('paper', None) if dataset == 'ogbn-mag' else None

        data = dataset[0].to(args.device)

        for model_name in args.models:
            if model_name not in supported_sets[dataset]:
                 print(f'Configuration of {dataset} + {model_name} not supported. Skipping.')
                 continue
            print(f'Evaluation bench for {model_name}:')
            for layers_amount in args.layers:
                print(f"Layers amount= {layers_amount}")
                for hidden_channels in args.num_hidden_channels:
                    print(f"Hidden features size= {layers_amount}")
                    params = {
                        'inputs_channels': dataset.num_features,
                        'hidden_channels': hidden_channels,
                        'output_channels': dataset.num_classes,
                        'num_relations': args.num_relations,
                        'num_layers': args.num_layers,
                    }
                    model = get_model(model_name, params)
                    for batch_size in args.eval_batch_sizes:

                        subgraph_loader = NeighborLoader(copy.copy(data),
                                                        num_neighbors=[-1],
                                                        input_nodes=mask,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        )

                        if args.pure_gnn:
                            i = 0
                            prebatched_samples = []
                            for batch in subgraph_loader:
                                if i == args.prebatched_samples:
                                    break
                                prebatched_samples.append(batch)
                                i += 1
                            subgraph_loader = prebatched_samples

                        start = default_timer()
                        model.inference(subgraph_loader,
                                        device=args.device,
                                        data = None if dataset == 'ogbn-mag' else data.x)
                        stop =  default_timer()
                        print(f'Batch size={batch_size} Inference time={start-stop}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')

    argparser.add_argument('--device', default='cpu', type=str)
    argparser.add_argument('--pure_gnn', action='store_false',
                           help="turn on pure gnn efficiency bench - firstly prepare batches")
    argparser.add_argument('--prebatched_samples', default=5 , typer=int,
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
    argparser.add_argument('--num-relations', default= [3], type=int)
    argparser.add_argument('--num-workers', default=0, type=int)

    args = argparser.parse_args()

    run(args)