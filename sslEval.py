"""
Evaluate models against baseline model
"""
import torch
import torch.nn as nn
import argparse
import yaml
from torch.utils.data import DataLoader
from sslGetData import loadRML22, IQDataset
from sslFineTuner import blHandler
from getModel import blModel
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description='Evaluate models against baseline model')
parser.add_argument('--data', type=str, default='./Data/RML22_TestDataset.pickle.16S',
                    help='Path to the data file')
parser.add_argument('--config', type=str, default='./Config/config.yaml',
                    help='Path to the config file')
parser.add_argument('--models', type=str, nargs='+', required=True,
                    help='Names of the models to evaluate as specified in the config file')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for evaluation')

args = parser.parse_args()


def build_models_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    models = config.get('models')
    model_dict = {}

    for model in models:
        in_dim = model.get('in_dim')
        model_name = model.get('name')
        feature_dim = model.get('feature_dim')

        if model.get('type') == 'bl':
            try:
                model_dict[model_name] = blModel(
                    input_dim=in_dim, feature_dim=feature_dim)
                model_dict[model_name].load_state_dict(
                    torch.load(model.get('output_path'), weights_only=True))
            except Exception:
                model_dict[model_name] = nn.Sequential(*list(blModel(
                    input_dim=in_dim, feature_dim=feature_dim).children()))
                model_dict[model_name].load_state_dict(
                    torch.load(model.get('output_path'), weights_only=True))
        elif model.get('type') == 'eev2':
            raise NotImplementedError

    return model_dict


def main():
    X, y, snrs = loadRML22(args.data)
    print(f"Evaluating on {X.shape[0]} examples")
    dataset = IQDataset(X, y, snrs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    accuracies = []
    for model_name in args.models:
        model = build_models_from_config(args.config).get(model_name)
        print("Evaluating model: ", model_name)
        # print("Model: ", model)
        print("Model has ", sum(p.numel()
              for p in model.parameters() if p.requires_grad), " parameters")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, acc = blHandler.infer(model, dataloader, device)
        accuracies.append(acc)

    # plot accuracies
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for acc, model_name in zip(accuracies, args.models):
        ax.plot(acc.keys(), acc.values(), 'o-', label=model_name)
    ax.set_xlabel('SNR')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid()

    plt.savefig('ModelComparison.png', dpi=300)


if __name__ == '__main__':
    main()
