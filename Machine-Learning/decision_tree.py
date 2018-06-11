from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf1 = DecisionTreeClassifier(min_samples_split=50)
clf2 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
clf2.fit(features_train, labels_train)

pred1 = clf1.predict(features_test)
pred2 = clf2.predict(features_test)

acc_min_samples_split_50 = accuracy_score(pred1, labels_test)
acc_min_samples_split_2 = accuracy_score(pred2, labels_test)

print( acc_min_samples_split_2 )
print( acc_min_samples_split_50 )