import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

tot_r_21 = 0
tot_score0 = 0
tot_mse = 0
i = 1
while i < 100:
    df = pd.DataFrame()

    rand = np.transpose(sorted(np.random.rand(100)))
    # print(rand)
    # plt.plot(rand)
    # plt.show()
    rand_new = np.array(rand)
    rand_new[[15, 30, 40, 55, 60, 73, 86, 90]] = np.nan

    a = np.array(range(1, 101))
    a = a.reshape(-1, 1)
    # print(rand_new)

    nan_array = np.isnan(rand_new)
    not_nan_array = ~ nan_array
    rand_cal = rand_new[not_nan_array]
    mean = np.mean(rand_cal)
    rand_new[[15, 30, 40, 55, 60, 73, 86, 90]] = mean
    # print(rand_new)

    rand_new = rand_new.reshape(-1, 1)
    # print(rand_new)

    lm = LinearRegression()
# Just predict the nan row
    lm.fit(a, rand_new)
    result = lm.predict(np.array([15, 30, 40, 55, 60, 73, 86, 90]).reshape(-1, 1))
    # print(result)

    newest = rand_new
    newest[[15, 30, 40, 55, 60, 73, 86, 90]] = result
    sns.residplot(x=rand, y=rand_new)
# plt.show()

# R-squared
    r_2 = sklearn.metrics.r2_score(rand[[15, 30, 40, 55, 60, 73, 86, 90]], result)
    print('The R-squared value is : ', r_2)

    r_21 = sklearn.metrics.r2_score(rand, rand_new)
    print('The R-squared value for the whole model is : ', r_21)

# predict whole model
    lm.fit(a, rand_new)
    new_result = lm.predict(a.reshape(-1, 1))

    score0 = lm.score(a, rand_new)
    print('score : ', score0)
    mse = metrics.mean_squared_error(rand_new, new_result)
    print('MSE score : ', mse)

    tot_r_21 += r_21
    tot_score0 += score0
    tot_mse += mse
    i = i + 1

print('The average r-squared : ', tot_r_21/99)
print('The average score : ', tot_score0/99)
print('The average mse : ', tot_mse/99)

sns.residplot(a, rand_new)
plt.figure()
sns.regplot(a, newest)
plt.show()
