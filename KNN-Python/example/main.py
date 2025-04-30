
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knn.knn import KNN

if __name__ == "__main__":
	X_train = [[1,2],[2,3],[3,1],[6,5],[7,7]]
	y_train = ['A','A','A','B','B']

	X_test = [[5,5]]

	model = KNN(k=3)
	model.fit(X_train,y_train)

	prediction = model.predict(X_test)

	print("predicted class :",prediction)