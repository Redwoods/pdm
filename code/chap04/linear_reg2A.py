import matplotlib.pylab as plt
from sklearn import linear_model

# 선형 회귀 모델을 생성한다. 
reg = linear_model.LinearRegression()

# 데이터는 파이썬의 리스트로 만들어도 되고 아니면 넘파이의 배열로 만들어도 됨
X = [[0], [1], [2]]		# 반드시 2차원으로 만들어야 함
y = [3, 3.5, 5.5]		# y = x + 3

# 학습을 시킨다. 
reg.fit(X, y)	

print(reg.coef_)		# 직선의 기울기
print(reg.intercept_) 	# 직선의 y-절편 
print(reg.score(X, y))  # R2 accuracy

print(reg.predict([[5]]))


# 학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(X)
from sklearn.metrics import r2_score
print(r2_score(y, y_pred))

# 학습 데이터와 y 값을 산포도로 그린다. 
plt.scatter(X, y, color='black')
# 학습 데이터와 예측값으로 선그래프로 그린다. 
# 계산된 기울기와 y 절편을 가지는 직선이 그려진다. 
plt.plot(X, y_pred, color='blue', linewidth=3)		
plt.show()

################  Refs   #####################
# https://losskatsu.github.io/machine-learning/sklearn/#%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95-%EB%82%98%EB%AC%B4
# https://wotres.tistory.com/entry/%EB%B6%84%EB%A5%98-%EC%84%B1%EB%8A%A5-%EC%B8%A1%EC%A0%95%ED%95%98%EB%8A%94%EB%B2%95-Accuracy-Precision-Recall-F1-score-ROC-AUC-in-python
##############################################