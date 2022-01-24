
import argparse
import os.path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch import optim

from data import get_train_data_loaders
from saint import SAINTModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/tmp/temp-mhkwon/Ednet-KT1/KT1-train.csv")
parser.add_argument("--max_seq", type=int, default=128)
parser.add_argument("--min_items", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--h_dim", type=int, default=512)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gpu", type=str, default="2")
args = parser.parse_args()

data_set_name = os.path.basename(args.data_path)
save_path = "./trained/saint-" + data_set_name[:-4] + "_d{}_l{}_dr{}_lr{}_b{}.pt".format(
                        args.h_dim, args.n_layers, args.dropout_prob, args.lr, args.batch_size)
print("model will be stored on:", save_path)

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
# Tensors created for attention have different types without this code.
torch.set_default_dtype(torch.float64)

train_loader, val_loader, total_ex = get_train_data_loaders(args)
total_cat = 7 # (1 ~ 7)
total_in = 2 # (O, X)

model = SAINTModel(embed_dim=args.h_dim, n_layers=args.n_layers, dropout_prob=args.dropout_prob,
                   max_seq=args.max_seq, n_skill=total_ex, n_part=total_cat).to(device)

criterion = nn.BCEWithLogitsLoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters())

# Training
for epoch in range(args.epochs):
    model.train()
    train_loss = []
    all_labels = []
    all_outs = []
    top_auc = 0.

    start_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs["qids"].to(device), inputs["part_ids"].to(device), inputs["correct"].to(device))
        mask = (inputs["qids"] != 0).to(device)
        labels = labels.to(device).float()

        output_mask = torch.masked_select(outputs, mask)
        label_mask = torch.masked_select(labels, mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        train_loss.append(loss_value)

        all_labels.extend(label_mask.view(-1).data.cpu().numpy())
        all_outs.extend(output_mask.view(-1).data.cpu().numpy())

        if i % 100 == 99:
            print("[%d, %5d] loss: %.3f, %.2f" %
                  (epoch + 1, i + 1, loss_value, time.time() - start_time))
            loss_value = 0

    train_auc = roc_auc_score(all_labels, all_outs)
    train_loss = np.mean(train_loss)

    # save best performing model
    if train_auc > top_auc:
        top_auc = train_auc
        torch.save(model, save_path)

    #####################################################################

    # test, evaluate with metrics
    model.eval()
    val_loss = []
    all_labels = []
    all_outs = []
    # `with torch.no_grad()` is necessary for saving memory.
    # Or it causes gpu memory allocation error.
    # Maybe it was due to memory overspending caused by cloning embedding
    # (from arshadshk's model).
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data

            outputs = model(inputs["qids"].to(device), inputs["part_ids"].to(device), inputs["correct"].to(device))
            mask = (inputs["qids"] != 0).to(device)

            outputs = torch.masked_select(outputs.squeeze(), mask)
            labels = torch.masked_select(labels.to(device), mask)
            loss = criterion(outputs.float(), labels.float())
            val_loss.append(loss.item())

            # calc auc, acc
            all_labels.extend(labels.view(-1).data.cpu().numpy())
            all_outs.extend(outputs.view(-1).data.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_outs)
        val_loss = np.mean(val_loss)

        print("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(
            epoch + 1, train_loss, train_auc, val_loss, val_auc, time.time() - start_time))



