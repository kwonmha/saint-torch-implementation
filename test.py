
import argparse

from sklearn.metrics import roc_auc_score
import torch

from data import get_test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

test_loader = get_test_data_loader(args)
model = torch.load(args.model_path)
model.eval()

all_labels = []
all_outs = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data

        outputs = model(inputs["qids"].to(device), inputs["part_ids"].to(device), inputs["correct"].to(device))
        mask = (inputs["qids"] != 0).to(device)

        outputs = torch.masked_select(outputs.squeeze(), mask)
        labels = torch.masked_select(labels.to(device), mask)

        # calc auc
        all_labels.extend(labels.view(-1).data.cpu().numpy())
        all_outs.extend(outputs.view(-1).data.cpu().numpy())

    auc = roc_auc_score(all_labels, all_outs)

    print("AUC - {:.4f} ".format(auc))
