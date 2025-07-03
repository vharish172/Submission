data = pd.read_csv("loan_access_dataset.csv")
data = data.dropna()
data['AgeGroup'] = [0 if i < 30 else 1 if i < 50 else 2 for i in data['Age']]

data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})
data['Race'] = data['Race'].map({'White':0, 'Black':1, 'Asian':2, 'Other':3})
data['Employment_Type'] = LabelEncoder().fit_transform(data['Employment_Type'])

temp = data.copy()
X = temp[['Gender','Race','Income','Credit_Score','Loan_Amount','Employment_Type','AgeGroup']]
Y = temp['Approved']

x1, x2, y1, y2 = train_test_split(X, Y, test_size=0.3)

for col in ['Income','Credit_Score','Loan_Amount']:
    x1[col] = (x1[col] - x1[col].mean()) / x1[col].std()
    x2[col] = (x2[col] - x2[col].mean()) / x2[col].std()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x1, y1)
preds = clf.predict(x2)
print("acc is ", accuracy_score(y2, preds))

m1 = data[data['Gender']==1]['Approved'].mean()
m0 = data[data['Gender']==0]['Approved'].mean()
print("diff:", m1-m0)
