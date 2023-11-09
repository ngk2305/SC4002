import random

file_path = 'train.csv'

with open(file_path, 'r') as file:
    lines = file.readlines()
    random.shuffle(lines)

with open(file_path, 'w') as file:
    file.writelines(lines)