import itertools
import pandas as pd

from sklearn import linear_model
from regressors import stats
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
### refer: https://regressors.readthedocs.io/en/latest/usage.html


def getCombination(n_components,iter_obj):
    return itertools.combinations(iter_obj,n_components)




def computeNcomponents(n,file_path=None):
    cols = []
    for i in range(1,n+1):
        cols.append("数据组合{}".format(i))
    cols.append("R2值")
    cols.append("R2_adj值")
    for i in range(1,n+1):
        cols.extend(["数据{}_coef".format(i),"数据{}_p".format(i),"数据{}_t".format(i)])
    cols.extend(["Intercept_coef","Intercept_p","intercept_t"])

    dataFrame = pd.read_excel(file_path)
    columns = list(dataFrame.columns)
    x_arry = columns[2:]
    y_arry = dataFrame['销售额(千元)']

    combines = getCombination(n, x_arry)  # default 3, we can change it as needed

    rows = []
    for c in combines:
        row = list(c)
        ols = linear_model.LinearRegression()
        # print(dataFrame.ix[:, c])
        ols.fit(dataFrame.ix[:, c], y_arry)

        p_values = stats.coef_pval(ols, dataFrame.ix[:, c],y_arry)  # return n+1 p_values, first one may be intercept value
        t_values = stats.coef_tval(ols, dataFrame.ix[:, c], y_arry)
        r2_adj = stats.adj_r2_score(ols, dataFrame.ix[:, c], y_arry)
        r2 = ols.score(dataFrame.ix[:, c], y_arry)
        x_coef = ols.coef_
        row.append(r2)
        row.append(r2_adj)
        for i in range(len(c)):  # format output
            row.append(x_coef[i])
            row.append(p_values[i + 1])
            row.append(t_values[i + 1])

        row.extend([ols.intercept_,p_values[0],t_values[0]])
        rows.append(row)
        # print(p_values,t_values,r_value)

    DF = pd.DataFrame(rows, columns=cols)
    return DF


def main():
    import  time
    start = time.clock()
    data_files = os.listdir(data_dir)
    components = [3,4] # we can change it manually
    writer = pd.ExcelWriter("output.xlsx")
    for f in data_files:
        for n in components:
            res_df = computeNcomponents(n,os.path.join(data_dir,f))
            res_df.to_excel(writer,sheet_name="{0}_{1}components.xlsx".format(f[:-5],n))

    writer.save()
    end =time.clock()
    print("time elapsed: {} seconds".format(end-start))


if __name__ == '__main__':
    # x = ['a','b','c','d']
    # li = getCombination(2,x)
    # for r in li:
    #     print(r)
    main()


