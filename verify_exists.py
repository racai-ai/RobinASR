import os


def remove_nonexists(path):
    with open(path, "r", encoding='utf-8') as file, open(path + "1", "w", encoding='utf-8') as out_file:
        for i, line in enumerate(file):
            path1, path2 = line.split(",")
            path2 = path2[:-1]

            if os.path.exists(path1) and os.path.exists(path2):
                if i == 0:
                    out_file.write("{},{}".format(path1, path2))
                else:
                    out_file.write("\n{},{}".format(path1, path2))
            else:
                print(path1, path2)


remove_nonexists("data/train_manifest.csv")
remove_nonexists("data/val_manifest.csv")