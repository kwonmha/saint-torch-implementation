
import argparse
import os.path
import time

from modelsummary import summary
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch import optim

from data import get_train_data_loaders
from models.saint import SaintModel, NoamOpt, NoamLR

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/var/tmp/mhkwon/riiid-kt1/KT1-train-50k.csv")
parser.add_argument("--max_seq", type=int, default=100)
parser.add_argument("--min_items", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--warmup_step", type=int, default=4000)
parser.add_argument("--h_dim", type=int, default=256)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gpu", type=str, default="1")
parser.add_argument("--noam", action="store_true", default=False)
parser.add_argument("--mynoam", action="store_true", default=False)
args = parser.parse_args()

data_set_name = os.path.basename(args.data_path)
save_path = "./trained/saint-" + data_set_name[:-4] + "_d{}_l{}_dr{}_lr{}_b{}_{}.pt".format(
                        args.h_dim, args.n_layers, args.dropout_prob, args.lr, args.batch_size, args.optimizer)
if args.noam:
    save_path = save_path[:-3] + "-noam.pt"
elif args.mynoam:
    save_path = save_path[:-3] + "-mynoam.pt"

print("model will be stored on:", save_path)

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Tensors created for attention have different types without this code.
# It was due to arshadshk's model design.
# torch.set_default_dtype(torch.float64)

train_loader, val_loader, total_ex = get_train_data_loaders(args)
total_cat = 7 # (1 ~ 7)
total_in = 2 # (O, X)

model = SaintModel(dim_model=args.h_dim, num_en=args.n_layers, num_de=args.n_layers,
                   heads_en=args.n_heads, heads_de=args.n_heads, dropout_prob=args.dropout_prob, seq_len=args.max_seq,
                   total_ex=total_ex, total_cat=total_cat, total_in=total_in, device=device).to(device)
# model summary
# input_size = (args.max_seq, 1)
# sample_input = torch.zeros(input_size, dtype=torch.long, device=device)
# print(summary(model, sample_input, sample_input, sample_input))

# sigmoid included in BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
criterion.to(device)

if args.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
elif args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.noam:
    optimizer = NoamOpt(args.h_dim, args.warmup_step, optimizer)
elif args.mynoam:
    scheduler = NoamLR(args.h_dim, args.warmup_step, optimizer)

print("start training")
# Training
top_auc = 0.
for epoch in range(args.epochs):
    model.train()
    train_loss = []
    all_labels = []
    all_outs = []

    start_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        exercise = inputs["qids"].to(device)
        part = inputs["part_ids"].to(device)
        response = inputs["correct"].to(device)

        optimizer.zero_grad()
        outputs = model(exercise, part, response)
        mask = (exercise != 0)
        labels = labels.to(device)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        # scheduler.step()
        loss_value = loss.item()
        train_loss.append(loss_value)

        output_mask = torch.masked_select(outputs, mask)
        label_mask = torch.masked_select(labels, mask)

        all_labels.extend(label_mask.view(-1).data.cpu().numpy())
        all_outs.extend(output_mask.view(-1).data.cpu().numpy())

        if i % 500 == 499:
            print("[%d, %5d] loss: %.3f, %.2f" %
                  (epoch + 1, i + 1, loss_value, time.time() - start_time))

    train_auc = roc_auc_score(all_labels, all_outs)
    train_loss = np.mean(train_loss)

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
            exercise = inputs["qids"].to(device)
            part = inputs["part_ids"].to(device)
            response = inputs["correct"].to(device)

            outputs = model(exercise, part, response)
            mask = (inputs["qids"] != 0).to(device)
            labels = labels.to(device)

            loss = criterion(outputs, labels.float())
            val_loss.append(loss.item())

            outputs = torch.masked_select(outputs, mask)
            labels = torch.masked_select(labels, mask)

            # calc auc, acc
            all_labels.extend(labels.view(-1).data.cpu().numpy())
            all_outs.extend(outputs.view(-1).data.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_outs)
        val_loss = np.mean(val_loss)

        # save best performing model
        print("val, top auc", val_auc, top_auc)
        if val_auc > top_auc:
            top_auc = val_auc
            print("saved best model")
            torch.save(model.state_dict(), save_path)

        print("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(
            epoch + 1, train_loss, train_auc, val_loss, val_auc, time.time() - start_time))



