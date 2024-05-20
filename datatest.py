import os
import csv
from itertools import permutations
import random

train_file = open('fashion-resize-pairs-train.csv', 'w', newline='')
test_file = open('fashion-resize-pairs-test.csv', 'w', newline='')


train_writer = csv.writer(train_file)
test_writer = csv.writer(test_file)


train_writer.writerow(['from', 'to', 'garment'])
test_writer.writerow(['from', 'to', 'garment'])

for (path, dir, files) in os.walk("./fashion"):
    if dir == []:
        new_path = path[2:]
        path_in_list = new_path.split('\\')

        top_related_garments = ['Jackets_Vests', 'Shirts_Polos', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Graphic_Tees']
        if (path_in_list[1] not in ['MEN', 'WOMEN']) or (path_in_list[2] not in top_related_garments):
            continue
        path_in_list[3] = path_in_list[3][:2] + path_in_list[3][3:]
        separated_files = {}
        for file in files:
            file = file[:4] + file[5:]
            prefix = file[:2]
            if prefix not in separated_files:
                separated_files[prefix] = []
            separated_files[prefix].append(file)
        for file_list in separated_files.values():
            file_front_name_list = list(map(lambda filename: filename.split(".")[0][4:], file_list))
            if 'flat' not in file_front_name_list:  # skip if 'flat' is missing
                continue
            file_front_without_flat = list([item for item in file_front_name_list if item != 'flat'])
            if len(file_front_without_flat) < 2:  # skip if there are fewer than two items, excluding 'flat'
                continue
            path_file_list = list(map(lambda filename: "".join(path_in_list) + filename, file_list))
            perm_input = [item for item in path_file_list if item[-8:-4] != 'flat']
            garment_element = [item for item in path_file_list if item[-8:-4] == 'flat'][0]
            perm_outputs = list(permutations(perm_input, 2))
            for perm_output in perm_outputs:
                if random.random() < 0.1:
                    test_writer.writerow(perm_output + (garment_element,))
                else:
                    train_writer.writerow(perm_output + (garment_element,))

train_file.close()
test_file.close()
