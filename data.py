
import gc
import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def get_train_data_loaders(args):
    train_df = pd.read_csv(args.data_path)
    train_df["question_id"] = train_df["question_id"].apply(lambda qid: int(qid[1:]))
    train_df = train_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

    dir_path = os.path.dirname(args.data_path)
    question_df = pd.read_csv(dir_path + "/questions.csv")
    question_df["question_id"] = question_df["question_id"].apply(lambda qid: int(qid[1:]))
    q2e_mapping = None
    # for saving embedding space
    # q2e_mapping = {qid: i for i, qid in enumerate(question_df.question_id.unique())}
    # n_q = question_df.question_id.nunique()  # n_data: 13169, max:18143
    n_q = question_df.question_id.max()

    merge_df = pd.merge(left=train_df, right=question_df, how="inner", on="question_id")

    group = merge_df[["user_id", "question_id", "user_answer", "correct_answer", "part"]]\
                .groupby("user_id")\
                .apply(lambda r: (r.question_id.values, r.user_answer.values==r.correct_answer.values,
                                  r.part.values, len(r.question_id)))

    ######### code for checking length distribution ############
    # lengths = [group[id][-1] for id in group.index]
    # from collections import Counter
    # length_count = Counter(lengths)
    # length_count = sorted(length_count.items())
    # length_count = {length:count for length, count in length_count}
    # length_df = pd.DataFrame(length_count.values(), index=length_count.keys())
    # length_df["cum"] = length_df[0].cumsum()
    ##########################################

    # test_data is already splitted during preprocessing.
    # train:val:test = 7:1:2
    train_data, val_data = train_test_split(group, test_size=1/8)
    del train_df, question_df, merge_df, group
    gc.collect()

    train_dataset = KT1Dataset(train_data, q2e_mapping, max_seq=args.max_seq, min_items=args.min_items)
    val_dataset = KT1Dataset(val_data, q2e_mapping, max_seq=args.max_seq, min_items=args.min_items)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False)

    return train_loader, val_loader, n_q


def get_test_data_loader(args):
    test_df = pd.read_csv(args.data_path)
    test_df["question_id"] = test_df["question_id"].apply(lambda qid: int(qid[1:]))
    test_df = test_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

    dir_path = os.path.dirname(args.data_path)
    question_df = pd.read_csv(dir_path + "/questions.csv")
    question_df["question_id"] = question_df["question_id"].apply(lambda qid: int(qid[1:]))

    merge_df = pd.merge(left=test_df, right=question_df, how="inner", on="question_id")

    test_data = merge_df[["user_id", "question_id", "user_answer", "correct_answer", "part"]] \
        .groupby("user_id") \
        .apply(lambda r: (r.question_id.values, r.user_answer.values == r.correct_answer.values,
                          r.part.values, len(r.question_id)))

    test_dataset = KT1Dataset(test_data, None, max_seq=args.max_seq, min_items=args.min_items)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False)

    return test_loader


class KT1Dataset(Dataset):
    def __init__(self, data, q2e_mapping, max_seq, min_items):
        super(KT1Dataset, self).__init__()
        self.max_seq = max_seq
        self.inputs = []

        # 모든 데이터를 list 형태의 self.inputs에 넣는다.
        for id in data.index:
            qids, correct, part_ids, length = data[id]
            # qids = [q2e_mapping[qid] for qid in qids]
            correct = [1 if c else 0 for c in correct]

            if len(qids) > max_seq:
                # 학습 데이터의 길이가 긴 경우 여러 번
                for start in range((len(qids) + max_seq - 1) // max_seq):
                    self.inputs.append((qids[start:start+max_seq], part_ids[start:start+max_seq],
                                       correct[start:start+max_seq]))
            elif min_items <= len(qids) < max_seq:
                self.inputs.append((qids, part_ids, correct))
            # len(qids) < min_items
            else:
                continue

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        qids, part_ids, correct = self.inputs[index]
        seq_len = len(qids)

        qid_tensor = np.zeros(self.max_seq, dtype=int)
        part_ids_tensor = np.zeros(self.max_seq, dtype=int)
        input_correct_tensor = np.zeros(self.max_seq, dtype=int)
        target_correct_tensor = np.zeros(self.max_seq, dtype=int)

        qid_tensor[-seq_len:] = qids
        part_ids_tensor[-seq_len:] = part_ids
        # 0, 1-> 1, 2 for rsponse embedding id
        input_correct_tensor[-seq_len+1:] = [x+1 for x in correct[:-1]]
        target_correct_tensor[-seq_len:] = correct

        input_tensor = {"qids": qid_tensor[1:],
                        "part_ids": part_ids_tensor[1:],
                        "correct": input_correct_tensor[:-1]}
        return input_tensor, target_correct_tensor[1:]


