import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde
import seaborn as sns
from lab1.utils import get_distribution, get_density, types, compute_trimmed_mean, get_func

num_bins = 15

def plot_hist():
    sizes = [10, 50, 1000]
    if not os.path.isdir("lab1/task1"):
        os.makedirs("lab1/task1")
    for name in types:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #plt.subplots_adjust(wspace=0.5)
        fig.suptitle(name)
        for i in range(len(sizes)):
            array = get_distribution(name, sizes[i])
            n, bins, patches = ax[i].hist(array, num_bins, density=True, edgecolor='blue', alpha=0.2)
            ax[i].plot(bins, get_density(name, bins), color='blue', linewidth=1)
            ax[i].set_title("n = " + str(sizes[i]))
        plt.savefig("lab1/task1/" + name + ".png")


def compute_stat_values():
    sizes = [10, 100, 1000]
    if not os.path.isdir("lab1/task2"):
        os.makedirs("lab1/task2")
    repeats = 1000
    for name in types:
        for size in sizes:
            mean, med, zr, zq, ztr = [], [], [], [], []
            for _ in range(repeats):
                array = get_distribution(name, size)
                mean.append(np.mean(array))
                med.append(np.median(array))
                zr.append((max(array) + min(array)) / 2)
                zq.append((np.quantile(array, 0.25) + np.quantile(array, 0.75)) / 2)
                ztr.append(compute_trimmed_mean(array))
            with open("lab1/task2/" + name + str(size) + ".tex", "w") as f:
                f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
                f.write("\\hline\n")
                f.write("& \\bar{x} & mediana & z_r & z_Q & z_tr & \\\\\n")
                f.write("E(z) & " + f"{np.around(np.mean(mean), decimals=4)} & "
                                    f"{np.around(np.mean(med), decimals=4)} & "
                                    f"{np.around(np.mean(zr), decimals=4)} & "
                                    f"{np.around(np.mean(zq), decimals=4)} & "
                                    f"{np.around(np.mean(ztr), decimals=4)} & \\\\\n")
                f.write("\\hline\n")
                f.write("D(z) & " + f"{np.around(np.mean(np.multiply(mean, mean)) - np.mean(mean) * np.mean(mean), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(med, med)) - np.mean(med) * np.mean(med), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(zr, zr)) - np.mean(zr) * np.mean(zr), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(zq, zq)) - np.mean(zq) * np.mean(zq), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(ztr, ztr)) - np.mean(ztr) * np.mean(ztr), decimals=4)} & \\\\\n")
                f.write("\\end{tabular}")


def box_plot_tukey():
    sizes = [20, 100]
    if not os.path.isdir("lab1/task3"):
        os.makedirs("lab1/task3")
    for name in types:
        plt.figure()
        arr_20 = get_distribution(name, sizes[0])
        arr_100 = get_distribution(name, sizes[1])
        for a in arr_20:
            if a <= -50 or a >= 50:
                arr_20 = np.delete(arr_20, list(arr_20).index(a))
        for a in arr_100:
            if a <= -50 or a >= 50:
                arr_100 = np.delete(arr_100, list(arr_100).index(a))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot((arr_20, arr_100), patch_artist=True, vert=False, labels=["n = 20", "n = 100"])
        for whisker in bp['whiskers']:
            whisker.set(color="black", alpha=0.3, linestyle=":", linewidth=1.5)
        for flier in bp['fliers']:
            flier.set(marker="D", markersize=4)
        for box in bp['boxes']:
            box.set_facecolor("#DDA0DD")
            box.set(alpha=0.6)
        for median in bp['medians']:
            median.set(color='black')
        plt.ylabel("n")
        plt.xlabel("X")
        plt.title(name)
        plt.savefig("lab1/task3/" + name + ".png")


def count_emissions():
    sizes = [20, 100]
    repeats = 1000
    data = []
    for name in types:
        count = 0
        for j in range(len(sizes)):
            for i in range(repeats):
                arr = get_distribution(name, sizes[j])
                min = np.quantile(arr, 0.25) - 1.5 * (np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
                max = np.quantile(arr, 0.75) + 1.5 * (np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
                for k in range(0, sizes[j]):
                    if arr[k] > max or arr[k] < min:
                        count += 1
            count /= repeats
            data.append(name + " n = $" + str(sizes[j]) + "$ & $" + str(np.around(count / sizes[j], decimals=3)) + "$")
    with open("lab1/task3/task3.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|}\n")
        f.write("\\hline\n")
        for row in data:
            f.write(row + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def empiric():
    sizes = [20, 60, 100]
    if not os.path.isdir("lab1/task4"):
        os.makedirs("lab1/task4")
    for name in types:
        if name == "poisson":
            interval = np.arange(6, 15, 1)
        else:
            interval = np.arange(-4, 4, 0.01)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.5)
        fig.suptitle(name)
        for j in range(len(sizes)):
            arr = get_distribution(name, sizes[j])
            for a in arr:
                if name == "poisson" and (a < 6 or a > 14):
                    arr = np.delete(arr, list(arr).index(a))
                elif name != "poisson" and (a < -4 or a > 4):
                    arr = np.delete(arr, list(arr).index(a))
            ax[j].set_title("n = " + str(sizes[j]))
            if name == "poisson":
                ax[j].step(interval, [get_func(name, x) for x in interval], color='#DDA0DD')
            else:
                ax[j].plot(interval, [get_func(name, x) for x in interval], color='#DDA0DD')
            if name == "poisson":
                arr_ex = np.linspace(6, 14)
            else:
                arr_ex = np.linspace(-4, 4)
            ecdf = ECDF(arr)
            y = ecdf(arr_ex)
            ax[j].step(arr_ex, y, color='blue', linewidth=0.5)
            plt.savefig("lab1/task4/" + name + "_emperic.png")


def kernel():
    sizes = [20, 60, 100]
    if not os.path.isdir("lab1/task4"):
        os.makedirs("lab1/task4")
    for name in types:
        if name == "poisson":
            interval = np.arange(6, 15, 1)
        else:
            interval = np.arange(-4, 4, 0.01)
        for j in range(len(sizes)):
            arr = get_distribution(name, sizes[j])
            for a in arr:
                if name == "poisson" and (a < 6 or a > 14):
                    arr = np.delete(arr, list(arr).index(a))
                elif name != "poisson" and (a < -4 or a > 4):
                    arr = np.delete(arr, list(arr).index(a))

            title = ["h = 1/2 * h_n", "h = h_n", "h = 2 * h_n"]
            bw = [0.5, 1, 2]
            fig, ax = plt.subplots(1, 3, figsize = (12, 4))
            plt.subplots_adjust(wspace = 0.5)
            for k in range(len(bw)):
                kde = gaussian_kde(arr, bw_method = 'silverman')
                h_n = kde.factor
                fig.suptitle(name + ", n = " + str(sizes[j]))
                ax[k].plot(interval, get_density(name, interval), color='blue', alpha=0.5, label='density')
                ax[k].set_title(title[k])
                sns.kdeplot(arr, ax = ax[k], bw = h_n * bw[k], label = 'kde', color = 'black')
                ax[k].set_xlabel('x')
                ax[k].set_ylabel('f(x)')
                ax[k].set_ylim([0, 1])
                if name == 'Poisson':
                    ax[k].set_xlim([6, 14])
                else:
                    ax[k].set_xlim([-4, 4])
                ax[k].legend()
            plt.savefig("lab1/task4/" + name + "_kernel_" + str(sizes[j]) + ".png")


if __name__ == "__main__":
    plot_hist()
    compute_stat_values()
    box_plot_tukey()
    count_emissions()
    empiric()
    kernel()