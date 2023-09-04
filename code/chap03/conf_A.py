import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
# str(digits)
# digits.data.shape
# digits.images.shape
# digits.target.shape
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2)

knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test)

##############################################################
# confusion matrix
# https://namu.wiki/w/%ED%98%BC%EB%8F%99%ED%96%89%EB%A0%AC
# https://blog.naver.com/tommybee/222663277170
##############################################################

disp = metrics.plot_confusion_matrix(knn, X_test, y_test)
plt.show()

print(f"{metrics.classification_report(y_test, y_pred)}\n")

#
# [DIY] n_neighbors = 3, 6일때의 혼동행렬을 구하고 비교하시오.
#


#
# New method to plot confusion_matrix
#
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# y_pred = mlp.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=mlp.classes_)
disp.plot(cmap='Blues')

plt.savefig('confusion_matrix.pdf')