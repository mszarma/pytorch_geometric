
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
}

def run(args: argparse.ArgumentParser) -> None:

    print("BENCHMARK STARTS")
    for dataset_name in args.datasets:
        print("Dataset: ", dataset_name)
        dataset = get_dataset(dataset_name, args.root)

        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None

        data = dataset[0].to(args.device)
        inputs_channels =  data.x_dict['paper'].size(-1) if dataset_name == 'ogbn-mag' else dataset.num_features
       # print(data.num_nodes.to_dict())
        # num_nodes_dict = {}
        # for node_type in data.node_types:
        #     num_nodes_dict[node_type] = data[node_type].num_nodes
        # print(num_nodes_dict)
        #print(data.adj_t_dict)
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
                    if dataset_name == 'ogbn-mag':
                        # data.adj_t_dict = {}
                        # for keys, (row, col) in data.edge_index_dict.items():
                        #     sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
                        #     adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
                        #     # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
                        #     if keys[0] != keys[2]:
                        #         data.adj_t_dict[keys] = adj.t()
                        #         data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
                        #     else:
                        #         data.adj_t_dict[keys] = adj.to_symmetric()
                        # data.edge_index_dict = None
                        hetero_params = {
                            'num_nodes_dict': data.num_nodes_dict,
                            'x_types': list(data.x_dict.keys()),
                            'edge_types': list(data.adj_t_dict.keys())
                        }
                        params = dict(params, **hetero_params)
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
                                        data = None if dataset == 'ogbn-mag' else data.x_dict)
                        stop =  default_timer()
                        print(f'Batch size={batch_size} Inference time={start-stop}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')

    argparser.add_argument('--device', default='cpu', type=str)
    argparser.add_argument('--pure_gnn', action='store_false',
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