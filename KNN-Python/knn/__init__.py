import numpy as np
from collections import Counter

class KNN:
	def __init__(self, k=3):
		self.k = k
		self.X_train = None
		self.y_train = None
	def fit(self,X,y):
		self.X_train = np.array(X)
		self.y_train = np.array(y)

	def predict(self ,X):
		X = np.array(X)
		predictions = [self._predict_single(x) for x in X]
		return np.array(predictions)

	def _predict_single(self,x):
		distance = [self._ecu_distance(x,x_train) for x_train in X_train]
		print(distance)
		k_indices = np.argsort(distance)[:self.k]
		k_nearest_labels = self.y_train[k_indices]

		most_common = Counter(k_nearest_labels).most_common(1)
		return most_common[0][0]

	def  _ecu_distance(self,x1,x2):
		return np.sqrt(np.sum((x1-x2)**2))


if __name__ == "__main__":
	X_train = [[1,2],[2,3],[3,1],[6,5],[7,7]]
	y_train = ['A','A','A','B','B']

	X_test = [[5,5]]

	model = KNN(k=3)
	model.fit(X_train,y_train)

	prediction = model.predict(X_test)

	print("predicted class :",prediction)