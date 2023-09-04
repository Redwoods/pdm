import matplotlib.pylab as plt
import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

# 당뇨병 데이터 세트를 적재한다. 
diabetes = datasets.load_diabetes()
str(diabetes)
print(diabetes.keys())
str(diabetes.DESCR)

##########################################
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
str(diabetes_X)
str(diabetes_y)
print(diabetes_X.data.shape, diabetes_y.data.shape)
##########################################
# 하나의 특징(BMI)만 추려내서 2차원 배열로 만든다. BMI 특징의 인덱스가 2이다.
diabetes_X_new = diabetes_X[:, np.newaxis, 2]
print(diabetes_X_new.data.shape )

# 학습 데이터와 테스트 데이터를 분리한다. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes_X_new, diabetes_y, test_size=0.1, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# regression coeff.s & score
regr.coef_, regr.intercept_
print(regr.score(X_train, y_train))

# 테스트 데이터로 예측해보자. 
y_pred = regr.predict(X_test) 
from sklearn import metrics
scores = metrics.r2_score(y_test, y_pred)  # r2_score for regression
print(scores)

# 실제 데이터와 예측 데이터를 비교해보자. 
# plt.plot(y_test, y_pred, '.')

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title(f"Test data: R2 = {np.round(scores,3)}")
plt.show()

#
# pandas
#
########################################
# 당뇨병 데이터 구조 => DataFrame
# Attribute Information:
#     - age     age in years
#     - sex
#     - bmi     body mass index
#     - bp      average blood pressure
#     - s1      tc, T-Cells (a type of white blood cells)
#     - s2      ldl, low-density lipoproteins
#     - s3      hdl, high-density lipoproteins
#     - s4      tch, thyroid stimulating hormone
#     - s5      ltg, lamotrigine
#     - s6      glu, blood sugar level
# - target  a quantitative measure of disease progression one year after baseline
# [참고] https://wikidocs.net/49981
#       https://suhyun72.tistory.com/entry/%EC%82%AC%EC%9D%B4%ED%82%B7%EB%9F%B0-scikit-learnsklearn-diabetes
#
########################################
df0 = datasets.load_diabetes(as_frame=True)
df0
type(df0)
df0.keys()
print(df0.DESCR)
df0.frame.shape
df0.frame.head()
# Make dataframe from data bunch
df=df0.frame
df.shape
df.info()
df.head()
df.columns
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6','target']
df[['bmi']].plot()
df[['bp']].plot()
# plt.show()
# 혈당치(glucose)와 target (당뇨병 진행도)
df[['s6']].plot()
df[['target']].plot()
plt.show()

# multiple plot
df[['bmi','bp','s6','target']].plot()
# df[['bmi','bp','s6']].plot()
plt.show()

# X and y
# 하나의 특징(s6: 혈당치)만 추려내서 2차원 배열로 만든다. BMI 특징의 인덱스가 2이다.
diabetes_X_new2=df.s6.values 
diabetes_X_new2=diabetes_X_new2[:,np.newaxis]
diabetes_X_new2.shape, diabetes_y.shape

X_train, X_test, y_train, y_test = train_test_split(diabetes_X_new2, diabetes_y, test_size=0.1, random_state=0)
X_train.shape
regr2 = linear_model.LinearRegression()
regr2.fit(X_train, y_train)
# regression coeff.s & score
regr2.coef_, regr2.intercept_
print(regr2.score(X_train, y_train))

# 테스트 데이터로 예측해보자. 
y_pred = regr2.predict(X_test) 

# 실제 데이터와 예측 데이터를 비교해보자. 
# plt.plot(y_test, y_pred, '.')

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, 'r', linewidth=3)
plt.show()

#
# 상관도표 (correlation) : df.corr()
#
import seaborn as sns
sns.heatmap(df.corr(), 
        xticklabels=df.columns,
        yticklabels=df.columns,
        vmin= -1, vmax=1.0) #, cmap='coolwarm')
plt.show()

sns.pairplot(df)
plt.show()

# s5(lamotrigine )와 target (당뇨병 진행도)
df[['s5']].plot()
df[['target']].plot()

sns.pairplot(df[['s5','target']])
plt.show()
# X and y
# 하나의 특징(s5: lamotrigine)만 추려내서 2차원 배열로 만든다. BMI 특징의 인덱스가 2이다.

diabetes_X_new3=df.s5.values 
diabetes_X_new3=diabetes_X_new3[:,np.newaxis]
diabetes_X_new3.shape, diabetes_y.shape

X_train, X_test, y_train, y_test = train_test_split(diabetes_X_new3, diabetes_y, test_size=0.1, random_state=0)
X_train.shape
regr3 = linear_model.LinearRegression()
regr3.fit(X_train, y_train)

# 테스트 데이터로 예측해보자. 
y_pred = regr3.predict(X_test) 

# 실제 데이터와 예측 데이터를 비교해보자. 
# plt.plot(y_test, y_pred, '.')

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, 'r', linewidth=3)
plt.show()

# regression coeff.s & score
regr3.coef_,regr3.intercept_
print(regr3.score(X_train, y_train))

##################################
# 비교 : bmi, s6, s5
##################################
print(regr.score(X_train, y_train))
print(regr2.score(X_train, y_train))
print(regr3.score(X_train, y_train))

sns.pairplot(df[['bmi','s5','s6','target']])
plt.show()
#
# {DIY} bp vs. target
#
