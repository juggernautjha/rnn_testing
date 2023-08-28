from sys import argv
from tqdm import tqdm
import random

def clean():
    fname = argv[1]
    target = argv[2]
    with open(fname, 'r') as myf:
        x = [i.strip() for i in myf.readlines()]
        orig_len = len(x)
        x = [f'{i.lower()}\n' for i in x if i.isalpha()]
        new_len = len(x)
        print(f"Successfully cleaned {fname}. Reduced number of rows from {orig_len} to {new_len}")
    targ = open(target, 'w')
    targ.writelines(x)
    targ.close()

def datagen():
    superset = [i for i in tqdm(open(argv[1]).readlines())]
    badset = [i for i in tqdm(open(argv[2]).readlines())]
    # print("Read")
    assert len(superset) >= len(badset), "Superset smaller than subset LMFAO"
    complement = [i for i in tqdm(superset) if i not in badset]
    print(f"Found {len(complement)} unique entries, saving them to {argv[3]}")
    targ = open(argv[3], 'w')
    targ.writelines(complement)
    targ.close()


def gendset(test_ratio : float = 0.25):
    slug = argv[1].split('.')[0]
    superset = [i for i in tqdm(open(argv[1]).readlines())]
    random.shuffle(superset)
    test_idx = int(test_ratio*len(superset))
    test, train = superset[:test_idx], superset[test_idx:]
    test_dump = open(f"{slug}_test.txt", "w")
    train_dump = open(f"{slug}_train.txt", "w")
    test_dump.writelines(test)
    train_dump.writelines(train)


if __name__ == '__main__':
    gendset()

    
