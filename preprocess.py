"""
This code combines `u*.csv` files into train and test csv files.
"""

import os
import os.path
import random

path = "/tmp/temp-mhkwon/Ednet-KT1/KT1-all"

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print("n files:", len(files))

# train+val : test = 8:2
test_files = random.sample(set(files), int(len(files) * 0.2))
train_files = set(files) - set(test_files)
print("split files")

# processing train_files
contents = ""
for i, file in enumerate(train_files):
    uid = file[1:-4]
    f = open(os.path.join(path, file))
    # print(uid)
    for j, line in enumerate(f.readlines()):
        if j > 0:
            contents += line[:-1] + "," + uid + "\n"
        else:
            first_line = line
    f.close()
    if i % 10000 == 9999:
        print(i+1)

first_line = first_line[:-1] + ",user_id\n"

contents = first_line + contents
train_file_name = "/tmp/temp-mhkwon/Ednet-KT1/KT1-train.csv"
f = open(train_file_name, "wt")
f.write(contents)
f.close()
print("saved training file")

# processing test_files
contents = ""
for i, file in enumerate(test_files):
    uid = file[1:-4]
    f = open(os.path.join(path, file))
    # print(uid)
    for j, line in enumerate(f.readlines()):
        if j > 0:
            contents += line[:-1] + "," + uid + "\n"
        else:
            first_line = line
    f.close()
    if i % 10000 == 9999:
        print(i+1)

first_line = first_line[:-1] + ",user_id\n"

contents = first_line + contents
test_file_name = "/tmp/temp-mhkwon/Ednet-KT1/KT1-test.csv"
f = open(test_file_name, "wt")
f.write(contents)
f.close()
print("saved training file.")
print("preprocessing completed.")
