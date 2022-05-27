
import argparse
import time

from sklearn.metrics import roc_auc_score
import torch

from data import get_test_data_loader
from models.saint import SaintModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, default="/var/tmp/mhkwon/riiid-kt1/KT1-test-all.csv")
parser.add_argument("--max_seq", type=int, default=100)
parser.add_argument("--min_items", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--h_dim", type=int, default=256)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--gpu", type=str, default="1")
args = parser.parse_args()

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

test_loader = get_test_data_loader(args)
print("loaded data")
model = SaintModel(dim_model=args.h_dim, num_en=args.n_layers, num_de=args.n_layers,
                   heads_en=8, heads_de=8, dropout_prob=args.dropout_prob, seq_len=args.max_seq,
                   total_ex=18143, total_cat=7, total_in=2, device=device)
model.load_state_dict(torch.load(args.model_path))
model.to(device)
# model = torch.load(args.model_path).to(device)
print("loaded model")
model.eval()

all_labels = []
all_outs = []
with torch.no_grad():
    start_time = time.time()
    for i, data in enumerate(test_loader):
        inputs, labels = data

        outputs = model(inputs["qids"].to(device), inputs["part_ids"].to(device), inputs["correct"].to(device))
        mask = (inputs["qids"] != 0).to(device)

        outputs = torch.masked_select(outputs.squeeze(), mask)
        labels = torch.masked_select(labels.to(device), mask)

        # calc auc
        all_labels.extend(labels.view(-1).data.cpu().numpy())
        all_outs.extend(outputs.view(-1).data.cpu().numpy())

        if i % 500 == 499:
            print("[%5d] %.2f" % (i + 1, time.time() - start_time))

    auc = roc_auc_score(all_labels, all_outs)

    print("AUC - {:.4f} ".format(auc))
