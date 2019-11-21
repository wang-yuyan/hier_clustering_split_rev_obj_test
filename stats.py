#get the stats of all results printed

import numpy as np

if __name__ == "__main__":
    split_file = "./split_objs_1000_census.txt"
    ckmm_file = "./ckmm_objs_1000_census.txt"
    stats_file = "./stats_1000_census.txt"
    split_f = open(split_file, "r")
    ckmm_f = open(ckmm_file, "r")
    stats_f = open(stats_file, "w")
    algorithms = ["bisecting k-means", "average linkage", "single linkage", "random", "upper bound"]

    stats_f.write("Algorithm / new obj mean / new obj var / ckmm mean / ckmm var \n")
    split_objs = []
    next(split_f)
    for line in split_f.readlines():
        line = line.strip().split(" ")
        for i in range(len(line)):
            line[i] = float(line[i])
        split_objs.append(line)
    split_objs = np.array(split_objs)
    split_f.close()

    split_mean = np.average(split_objs, axis=0)
    split_var = np.std(split_objs, axis=0)

    ckmm_objs = []
    next(ckmm_f)
    for line in ckmm_f.readlines():
        line = line.strip().split(" ")
        for i in range(len(line)):
            line[i] = float(line[i])
        ckmm_objs.append(line)
    ckmm_objs = np.array(ckmm_objs)
    ckmm_f.close()

    ckmm_mean = np.average(ckmm_objs, axis=0)
    ckmm_var = np.std(ckmm_objs, axis=0)

    for i in range(len(algorithms)):
        stats_f.write("{} {} {} {} {} \n".format(algorithms[i], str(split_mean[i]), str(split_var[i]), str(ckmm_mean[i]),  str(ckmm_var[i])))

    stats_f.close()


