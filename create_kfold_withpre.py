import os
import sys
import time
import random
import math


k = int(sys.argv[1])
folder = sys.argv[2]
out = sys.argv[3]
prefix = sys.argv[4]
req_prefix = sys.argv[4]


all_data = os.listdir(folder)
all_data = [x for x in all_data if req_prefix in x]
random.shuffle(all_data)
all_set = set(all_data)

foldsize = len(all_data) // k

for i in range(k-1):
    val = all_data[i*foldsize:min(len(all_data),i*foldsize+foldsize)]
    val_set = set(val)

    train = list(all_set - val_set)

    print('fold ', i+1)
    print('val ', len(val), ' train ', len(train))

    val_text_name = prefix + "_val_fold" + str(i+1) + ".txt"
    train_text_name = prefix + "_train_fold" + str(i+1) + ".txt"

    fval = open(val_text_name, 'w')
    ftra = open(train_text_name, 'w')

    fval.write(repr(val))
    fval.close()

    ftra.write(repr(train))
    ftra.close()


i = k-1
val = all_data[i*foldsize:]
val_set = set(val)

train = list(all_set - val_set)

print('fold ', i+1)
print('val ', len(val), ' train ', len(train))

val_text_name = prefix + "_val_fold" + str(i+1) + ".txt"
train_text_name = prefix + "_train_fold" + str(i+1) + ".txt"

fval = open(val_text_name, 'w')
ftra = open(train_text_name, 'w')

fval.write(repr(val))
fval.close()

ftra.write(repr(train))
ftra.close()
