"""
Evaluate models against baseline model
"""
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from sslGetData import loadRML22, IQDataset
from sslFineTuner import blHandler
from getModel import blModel


parser = argparse.ArgumentParser(
    description='Evaluate models against baseline model')
parser.add_argument('--data', type=str, default='./Data/RML22_TestDataset.pickle.16S',
                    help='Path to the data file')
parser.add_argument('--config', type=str, default='./Config/config.yaml',
                    help='Path to the config file')
parser.add_argument('--models', type=str, nargs='+', required=True,
                    help='Names of the models to evaluate as specified in the config file')

args = parser.parse_args()


class buildModelsfromConfig:
    def __init__(self, configPath):
        self.configPath = configPath
        self.modelDict = self.buildModels()

    def buildModels(self):
        with open(self.configPath, 'r') as file:
            config = yaml.safe_load(file)
        modelDict = {}

        for model in config['models']:
            modelDict[model['name']] = blModel(model)
        return modelDict


def main():
    X, y, snrs = loadRML22(args.data)
    dataset = IQDataset(X, y, snrs)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=False, num_workers=4)
    bl_model = buildModelsfromConfig(args.config).get('blModel')
    evaluationModels = []
    for model in args.models:
        evaluationModels.append(buildModelsfromConfig(args.config).get(model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, acc = blHandler.infer(bl_model, dataloader, device)
