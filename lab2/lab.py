import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as stats
from scipy.stats import chi2, t, norm, moment
from scipy.optimize import minimize
import os
import math


def scattering_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def save_ellipses(size, ros):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    fig, ax = plt.subplots(1, 3)
    size_str = "n = " + str(size)
    titles = [size_str + r', $ \rho = 0$', size_str + r', $\rho = 0.5 $', size_str + r', $ \rho = 0.9$']
    for i in range(len(ros)):
        sample = np.random.multivariate_normal([0, 0], [[1.0, ros[i]], [ros[i], 1.0]], size)
        x, y = sample[:, 0], sample[:, 1]
        scattering_ellipse(x, y, ax[i], edgecolor='pink')
        ax[i].grid()
        ax[i].scatter(x, y, s=5)
        ax[i].set_title(titles[i])
    fig.savefig('task1_data/' + str(size) + '.png', dpi=200)


def save_ellipses_for_mix(sizes):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    fig, ax = plt.subplots(1, 3)
    for i in range(len(sizes)):
        sample = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], sizes[i]) + \
                 0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], sizes[i])
        x, y = sample[:, 0], sample[:, 1]
        scattering_ellipse(x, y, ax[i], edgecolor='pink')
        ax[i].grid()
        ax[i].scatter(x, y, s=5)
        ax[i].set_title('n = ' + str(sizes[i]))
    fig.savefig('task1_data/mix.png', dpi=200)


def quadrant_coefficient(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(size):
        if x[i] >= med_x and y[i] >= med_y:
            n[1] += 1
        elif x[i] < med_x and y[i] >= med_y:
            n[2] += 1
        elif x[i] < med_x and y[i] < med_y:
            n[3] += 1
        elif x[i] >= med_x and y[i] < med_y:
            n[4] += 1
    return (n[1] + n[3] - n[2] - n[4]) / size


def get_coefficients(size, ro, repeats):
    pearson, quadrant, spearman = [], [], []
    for i in range(repeats):
        sample = np.random.multivariate_normal([0, 0], [[1, ro], [ro, 1]], size)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spearman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_coefficient(x, y))
    return pearson, spearman, quadrant


def get_coefficients_for_mix(size, repeats):
    pearson, quadrant, spearman = [], [], []
    for i in range(repeats):
        sample = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size) + \
                 0.1 * np.random.multivariate_normal([0, 0], [[10, 0.9], [0.9, 10]], size)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spearman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_coefficient(x, y))
    return pearson, spearman, quadrant


def save_table(ros, size):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    rows = []
    for ro in ros:
        p, s, q = get_coefficients(size, ro, 1000)
        rows.append(['$\\rho = ' + str(ro) + '~(\\ref{ro})$', '$r ~(\\ref{r})$',
                     '$r_Q ~(\\ref{rQ})$', '$r_S ~(\\ref{rS})$'])
        rows.append(['$E(z)$', np.around(np.mean(p), decimals=3),
                     np.around(np.mean(s), decimals=3),
                     np.around(np.mean(q), decimals=3)])
        rows.append(['$E(z^2)$', np.around(np.mean(np.asarray([el * el for el in p])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in s])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in q])), decimals=3)])
        rows.append(['$D(z)$', np.around(np.std(p), decimals=3),
                     np.around(np.std(s), decimals=3),
                     np.around(np.std(q), decimals=3)])
    with open("task1_data/" + str(size) + ".tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def save_table_for_mix(sizes):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    rows = []
    for size in sizes:
        p, s, q = get_coefficients_for_mix(size, 1000)
        rows.append(['$n = ' + str(size) + '$', '$r ~(\\ref{r})$',
                     '$r_Q ~(\\ref{rQ})$', '$r_S ~(\\ref{rS})$'])
        rows.append(['$E(z)$', np.around(np.mean(p), decimals=3),
                     np.around(np.mean(s), decimals=3),
                     np.around(np.mean(q), decimals=3)])
        rows.append(['$E(z^2)$', np.around(np.mean(np.asarray([el * el for el in p])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in s])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in q])), decimals=3)])
        rows.append(['$D(z)$', np.around(np.std(p), decimals=3),
                     np.around(np.std(s), decimals=3),
                     np.around(np.std(q), decimals=3)])
    with open("task1_data/mix.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def task1():
    sizes = [20, 60, 100]
    ros = [0, 0.5, 0.9]

    for size in sizes:
        save_ellipses(size, ros)
        save_table(ros, size)

    save_ellipses_for_mix(sizes)
    save_table_for_mix(sizes)


def get_least_squares_coefs(x, y):
    xy_m = np.mean(np.multiply(x, y))
    x_m = np.mean(x)
    x_2_m = np.mean(np.multiply(x, x))
    y_m = np.mean(y)
    b1_mnk = (xy_m - x_m * y_m) / (x_2_m - x_m * x_m)
    b0_mnk = y_m - x_m * b1_mnk

    return b0_mnk, b1_mnk


def abs_dev_val(b_arr, x, y):
    return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))


def get_linear_approx_coefs(x, y):
    init_b = np.array([0, 1])
    res = minimize(abs_dev_val, init_b, args=(x, y), method='COBYLA')

    return res.x


def draw(lsm_0, lsm_1, lam_0, lam_1, x, y, title, fname):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='pink', s=6, label='Выборка')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='pink', label='МНК')
    ax.plot(x, y_lam, color='red', label='МНМ')
    ax.plot(x, y_real, color='orange', label='Модель')
    ax.set(xlabel='X', ylabel='Y',
       title=title)
    ax.legend()
    ax.grid()
    fig.savefig(fname + '.png', dpi=200)


def task2():
    x = np.arange(-1.8, 2.1, 0.2)
    eps = np.random.normal(0, 1, size=20)
    # y = 2 + 2x + eps
    y = np.add(np.add(np.full(20, 2), x * 2), eps)
    # y2 with noise
    y2 = np.add(np.add(np.full(20, 2), x * 2), eps)
    y2[0] += 10
    y2[19] += -10

    if not os.path.exists("task2_data"):
        os.mkdir("task2_data")

    lsm_0, lsm_1 = get_least_squares_coefs(x, y)
    lam_0, lam_1 = get_linear_approx_coefs(x, y)

    lsm_02, lsm_12 = get_least_squares_coefs(x, y2)
    lam_02, lam_12 = get_linear_approx_coefs(x, y2)

    with open("task2_data/no_noise.tex", "w") as f:
        f.write("\\begin{enumerate}\n")
        f.write("\\item МНК, без возмущений:\n")
        f.write("$\hat{a}\\approx " + str(lsm_0) + "$, $\hat{b}\\approx " + str(lsm_1) + "$\n")
        f.write("\\item МНМ, без возмущений:\n")
        f.write("$\hat{a}\\approx " + str(lam_0) + "$, $\hat{b}\\approx " + str(lam_1) + "$\n")
        f.write("\\end{enumerate}\n")

    with open("task2_data/noise.tex", "w") as f:
        f.write("\\begin{enumerate}\n")
        f.write("\\item МНК, с возмущенями:\n")
        f.write("$\hat{a}\\approx " + str(lsm_02) + "$, $\hat{b}\\approx " + str(lsm_12) + "$\n")
        f.write("\\item МНМ, с возмущенями:\n")
        f.write("$\hat{a}\\approx " + str(lam_02) + "$, $\hat{b}\\approx " + str(lam_12) + "$\n")
        f.write("\\end{enumerate}\n")

    draw(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Выборка без возмущений', 'task2_data/no_noise')
    draw(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Выборка с возмущенями', 'task2_data/noise')


def get_cdf(x, mu, sigma):
    return stats.norm.cdf(x, mu, sigma)


def normal():
    sz = 100
    samples = np.random.normal(0, 1, size=sz)
    mu_c = np.mean(samples)
    sigma_c = np.std(samples)
    k = 7
    borders = np.linspace(mu_c - 3, mu_c + 3, num=(k - 1))

    p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
    for i in range(len(borders) - 1):
        val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))

    print(f"mu: " + str(np.around(mu_c, decimals=2)) + ", sigma: " + str(np.around(sigma_c, decimals=2)))

    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))

    res_arr = np.divide(np.multiply((n_arr - p_arr * 100), (n_arr - p_arr * 100)), p_arr * 100)

    intervals = [('$-\inf$', str(np.around(borders[0], decimals=2)))]
    for i in range(len(borders) - 1):
        intervals.append((str(np.around(borders[i], decimals=2)), str(np.around(borders[i + 1], decimals=2))))
    intervals.append((str(np.around(borders[-1], decimals=2)), '$+\inf$'))
    rows = [[i + 1, (intervals[i][0] + ', ' + intervals[i][1]),
                   "%.2f" % n_arr[i],
                   "%.4f" % p_arr[i],
                   "%.2f" % (sz * p_arr[i]),
                   "%.2f" % (n_arr[i] - sz * p_arr[i]),
                   "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
    rows.insert(0, ['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', '$np_i$', '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
    rows.append(['$\sum$', '$-$', "%.2f" % np.sum(n_arr), "%.4f" % np.sum(p_arr), "%.2f" % (sz * np.sum(p_arr)),
        "%.2f" % (np.sum(n_arr) - sz * np.sum(p_arr)), "%.2f" % np.sum(res_arr)])
    with open("task3_data/task3_normal.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def laplace():
    sz = 20
    samples = np.random.laplace(0, 1 / math.sqrt(2), size=sz)
    mu_c = np.mean(samples)
    sigma_c = np.std(samples)
    k = 7
    borders = np.linspace(mu_c - 3, mu_c + 3, num=(k - 1))

    p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
    for i in range(len(borders) - 1):
        val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))

    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))

    res_arr = np.divide(np.multiply((n_arr - p_arr * sz), (n_arr - p_arr * sz)), p_arr * sz)

    intervals = [('$-\inf$', str(np.around(borders[0], decimals=2)))]
    for i in range(len(borders) - 1):
        intervals.append((str(np.around(borders[i], decimals=2)), str(np.around(borders[i + 1], decimals=2))))
    intervals.append((str(np.around(borders[-1], decimals=2)), '$+\inf$'))
    rows = [[i + 1, (intervals[i][0] + ', ' + intervals[i][1]),
                   "%.2f" % n_arr[i],
                   "%.4f" % p_arr[i],
                   "%.2f" % (sz * p_arr[i]),
                   "%.2f" % (n_arr[i] - sz * p_arr[i]),
                   "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
    rows.insert(0, ['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', '$np_i$', '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
    rows.append(['$\sum$', '$-$', "%.2f" % np.sum(n_arr), "%.4f" % np.sum(p_arr), "%.2f" % (sz * np.sum(p_arr)),
        "%.2f" % (np.sum(n_arr) - sz * np.sum(p_arr)), "%.2f" % np.sum(res_arr)])
    with open("task3_data/task3_laplace.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def task3():
    if not os.path.exists("task3_data"):
        os.mkdir("task3_data")
    normal()
    laplace()


gamma = 0.95
alpha = 1 - gamma


def get_student_mo(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    t_a = t.ppf(1 - alpha / 2, n - 1)
    q_1 = med - s * t_a / np.sqrt(n - 1)
    q_2 = med + s * t_a / np.sqrt(n - 1)
    return q_1, q_2


def get_chi_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    q_1 =  s * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    q_2 = s * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
    return q_1, q_2


def get_as_mo(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    q_1 = med - s * u / np.sqrt(n)
    q_2 = med + s * u / np.sqrt(n)
    return q_1, q_2


def get_as_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    m4 = moment(samples, 4)
    e = m4 / (s * s * s * s)
    U = u * np.sqrt((e + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2


def task4():
    if not os.path.isdir("task4_data"):
        os.makedirs("task4_data")

    samples20 = np.random.normal(0, 1, size=20)
    samples100 = np.random.normal(0, 1, size=100)

    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace = 0.5)
    ax[0].set_ylim([0,1])
    ax[0].hist(samples20, 10, density = 1, edgecolor = 'black', color='pink')
    ax[0].set_title('N(0,1) hist, n = 20')
    ax[1].set_ylim([0,1])
    ax[1].hist(samples100, 10, density = 1, edgecolor = 'black', color='pink')
    ax[1].set_title('N(0,1) hist, n = 100')
    plt.savefig('task4_data/hist.png')

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(wspace = 0.5, hspace = 1)

    student_20 = get_student_mo(samples20, alpha)
    student_100 = get_student_mo(samples100, alpha)

    chi_20 = get_chi_sigma(samples20, alpha)
    chi_100 = get_chi_sigma(samples100, alpha)

    ax[0][0].plot([student_20[0], student_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'm interval, n = 20')
    ax[0][0].plot([student_100[0], student_100[1]], [0.6, 0.6], color='pink', marker = '.', linewidth = 1, label = 'm interval, n = 100')
    ax[0][0].set_ylim([0,1])
    ax[0][0].set_title('Classic approach')
    ax[0][0].legend()

    ax[0][1].plot([chi_20[0], chi_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'sigma interval, n = 20')
    ax[0][1].plot([chi_100[0], chi_100[1]], [0.6, 0.6], color='pink', marker = '.', linewidth = 1, label = 'sigma interval, n = 100')
    ax[0][1].set_ylim([0,1])
    ax[0][1].set_title('Classic approach')
    ax[0][1].legend()

    print(f"Классический подход:\n"
        f"n = 20 \n"
        f"\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
        f"n = 100 \n"
        f"\t m: " + str(student_100) + " \t sigma: " + str(chi_100) + "\n")

    as_mo_20 = get_as_mo(samples20, alpha)
    as_mo_100 = get_as_mo(samples100, alpha)

    as_d_20 = get_as_sigma(samples20, alpha)
    as_d_100 = get_as_sigma(samples100, alpha)

    ax[1][0].plot([as_mo_20[0], as_mo_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'm interval, n = 20')
    ax[1][0].plot([as_mo_100[0], as_mo_100[1]], [0.6, 0.6], color='pink', marker = '.', linewidth = 1, label = 'm interval, n = 100')
    ax[1][0].set_ylim([0,1])
    ax[1][0].set_title('Asymptotic approach')
    ax[1][0].legend()

    ax[1][1].plot([as_d_20[0], as_d_20[1]], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'sigma interval, n = 20')
    ax[1][1].plot([as_d_100[0], as_d_100[1]], [0.6, 0.6], color='pink', marker = '.', linewidth = 1, label = 'sigma interval, n = 100')
    ax[1][1].set_ylim([0,1])
    ax[1][1].set_title('Asymptotic approach')
    ax[1][1].legend()

    plt.savefig('task4_data/intervals.png')

    print(f"Асимптотический подход:\n"
        f"n = 20 \n"
        f"\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
        f"n = 100 \n"
        f"\t m: " + str(as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")

if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()