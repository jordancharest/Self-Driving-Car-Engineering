from sklearn.svm import SVC






X = [[0,0], [1,1]]
y = [0,1]


if __name__ == "__main__": 
    classifier = SVC(kernel='linear)
    classifier.fit(X,y)
    
    classifier.predict([[2,2]])
    
    