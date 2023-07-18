import numpy as np
import pandas as pd
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

filepath = "20230322_All_replicates.csv"


def venn_(filepath):
    file = pd.read_csv(filepath)

    def calcu_EF(post, pre, file, sum_post=None, sum_pre=None):
        # file：df；post,pre:col names
        if sum_pre is None and sum_post is None:
            sum_post = np.sum(file[post])
            sum_pre = np.sum(file[pre])
        a = (file[pre] + 0.375) * sum_post
        b = (file[post] + 0.375) * sum_pre
        w = sum_post / sum_pre
        values = b / a
        return values

    num = 0.05 * len(file)

    print(file.columns)

    list_ = [['XF_RPI4_1', 'XF_RPI7_1', 'XF_RPI10_1', ],
             ['XF_RPI5_1', 'XF_RPI8_1', 'XF_RPI11_1'],
             ['XF_RPI6_1', 'XF_RPI9_1', 'XF_RPI12_1']]

    names = [["py1355 sample 1", "py1355 sample 2", "py1355 sample 3"],
             ["py1361 sample 1", "py1361 sample 2", "py1361 sample 3"],
             ["INSR sample 1", "INSR sample 2", "INSR sample 3"]]

    for post in ['XF_RPI4_1', 'XF_RPI7_1', 'XF_RPI10_1',
                 'XF_RPI5_1', 'XF_RPI8_1', 'XF_RPI11_1', 'XF_RPI6_1', 'XF_RPI9_1',
                 'XF_RPI12_1', ]:
        file[f"EF{post}"] = calcu_EF(post, "S1", file)

    list2 = [['EFXF_RPI4_1', 'EFXF_RPI7_1', 'EFXF_RPI10_1', ],
             ['EFXF_RPI5_1', 'EFXF_RPI8_1', 'EFXF_RPI11_1'],
             ['EFXF_RPI6_1', 'EFXF_RPI9_1', 'EFXF_RPI12_1']]

    def venn_3_subset(num, setlist, namelist, outname):
        for i in range(3):
            num = int(num)
            a = set(file.nlargest(num, setlist[i][0]).index)
            b = set(file.nlargest(num, setlist[i][1]).index)
            c = set(file.nlargest(num, setlist[i][2]).index)

            venn3([a, b, c], set(namelist[i]))
            plt.savefig(f"HYR-INSR5/figs/{num}Venn-{outname}{i}.png", dpi=300, transparent=True)
            plt.show()

    venn_3_subset(num, list_, names, "counts")
    venn_3_subset(num, list2, names, "EF")
