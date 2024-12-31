import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier




#====================    TITANIC PREDICTION    ====================#



random_state = 55

mydata_train = pd.read_csv("train.csv")
df1 = pd.DataFrame(mydata_train)
mydata_test = pd.read_csv("test.csv")
df2 = pd.DataFrame(mydata_test)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)



#------------------  Prepare Train Data  -----------------#



# convert values to numeric (nan if not digit)
df1_copy = df1.copy()
df1_copy['Ticket'] = pd.to_numeric(df1_copy['Ticket'], errors='coerce')
ticket_list = []


limit_dict1 = {'lim1': df1_copy['Ticket'] < 1000,
              'lim2': (df1_copy['Ticket'] < 10000) & (df1_copy['Ticket'] >= 1000),
              'lim3': (df1_copy['Ticket'] < 28000) & (df1_copy['Ticket'] >= 10000),
              'lim4': (df1_copy['Ticket'] < 100000) & (df1_copy['Ticket'] >= 28000),
              'lim5': (df1_copy['Ticket'] < 200000) & (df1_copy['Ticket'] >= 100000),
              'lim6': (df1_copy['Ticket'] < 300000) & (df1_copy['Ticket'] >= 200000),
              'lim7': df1_copy['Ticket'] >= 300000}


# survival rates based on ticket numbers
for i in limit_dict1:
    key = limit_dict1[i]
    batch = df1_copy[key]
    unique,count = np.unique(batch['Survived'], return_counts=True)
    if i == 'lim1':
        survive = 0
    else:
        survive = count[1]/sum(count)
    ticket_list.append(round(survive,4))


# replace tickets(of all digits) with survival rate
df1['Ticket'] = df1['Ticket'].apply(lambda x: int(x) if str(x).isdigit() else x)
for (a,b) in zip(limit_dict1.values(), ticket_list):
    df1.loc[df1['Ticket'].apply(lambda x: str(x).isdigit()) & a, 'Ticket'] = b


df1['Ticket'] = df1['Ticket'].apply(lambda x: str(x))
lis1 = []
ind = 0


# replace alphanumeric tickets with only the alpha part 
# and append to list
for i in df1['Ticket']:
    if i[0].isalpha():
        try:
            space = i.rindex(' ')
            text = i[:space]
            lis1.append(text)
            df1.iat[ind,8] = text
        except:
            lis1.append(i)
    ind += 1


lis1 = sorted(lis1)
new1 = []
first = lis1[0]


# remove duplicates in list
while True:
    new1.append(first)
    counter = lis1.count(first)
    lis1 = lis1[counter:]
    if lis1 == []:
        break
    first = lis1[0]


# replace alpha part with its respective survival rate
for i in new1:
    filtered_rows1 = df1[df1['Ticket'] == i]
    unique, count = np.unique(filtered_rows1['Survived'], return_counts=True)
    if len(unique) == 1 and unique[0] == 0:
        survive = 0
    elif len(unique) == 1 and unique[0] == 1:
        survive = 1
    else:
        survive = count[1]/sum(count)
    survive = round(survive, 4)
    df1.loc[df1['Ticket'] == i,'Ticket'] = survive


honorifics = ['Miss.','Ms.','Mrs.','Master.',
              'Mr.','Rev.','Dr.','Mme.','Don.',
              'Major.','Jonkheer.','Col.','Mlle.',
              'Countess.','Lady.','Sir.','Capt.']


# find mean age based on honorifics and fill null vals
for i in honorifics:
    filtered_rows = df1[df1['Name'].str.contains(i, case=False, na=False)]
    age_mean = round(filtered_rows.loc[:, 'Age'].mean(), 4)
    df1.loc[(df1['Name'].str.contains(i)) & (df1['Age'].isna()), 'Age'] = age_mean


grouped_data = df1.groupby(['Embarked', 'Survived']).size().unstack(fill_value=0)
grouped_data.plot(kind='bar', stacked=False)
plt.show()


df1.fillna({'Embarked':'S'}, inplace=True)


# drop unneeded cols or cols with many null vals
# one-hot-code non-binary values
try:
    df1 = df1.drop(['PassengerId','Name','Cabin'], axis=1)
    cat_variables = ['Pclass',
                     'Sex',
                     'Embarked']
    
    df1 = pd.get_dummies(data=df1,
                         prefix=cat_variables,
                         columns=cat_variables,
                         dtype=int)
except:
    pass



#----------------------  Train Data  ---------------------#



X1 = df1.iloc[:,1:]
y = df1.iloc[:,0]


# normalize data
X1 = preprocessing.StandardScaler().fit(X1).transform(X1)
X_train,X_test,y_train,y_test = train_test_split(X1,y,train_size=0.8,random_state=random_state)

model = RandomForestClassifier()

model.fit(X_train,y_train)
predictions1 = model.predict(X_test)
accuracy = accuracy_score(predictions1,y_test)
precision = precision_score(predictions1,y_test)
recall = recall_score(predictions1,y_test)

print("RandomForest")
print(accuracy)
print(precision)
print(recall)



#--------------------  Train Test Data  ------------------#



df2['Ticket'] = df2['Ticket'].astype(str)

ind = 0

# replace alphanumeric vals with the numeric part only
for i in df2['Ticket']:
    if i[0].isalpha():
        try:
            space = i.rindex(' ')
            num = i[space+1:]
            df2.iat[ind,7] = num
        except:
            pass
    ind += 1


df2_copy = df2.copy()
df2_copy['Ticket'] = pd.to_numeric(df2_copy['Ticket'], errors='coerce')


limit_dict2 = {'lim1': df2_copy['Ticket'] < 1000,
              'lim2': (df2_copy['Ticket'] < 10000) & (df2_copy['Ticket'] >= 1000),
              'lim3': (df2_copy['Ticket'] < 28000) & (df2_copy['Ticket'] >= 10000),
              'lim4': (df2_copy['Ticket'] < 100000) & (df2_copy['Ticket'] >= 28000),
              'lim5': (df2_copy['Ticket'] < 200000) & (df2_copy['Ticket'] >= 100000),
              'lim6': (df2_copy['Ticket'] < 300000) & (df2_copy['Ticket'] >= 200000),
              'lim7': df2_copy['Ticket'] >= 300000}


df2['Ticket'] = df2['Ticket'].apply(lambda x: float(x) if str(x).isdigit() else x)


# replace tickets(of all digits) with survival rate
for condition, value in zip(limit_dict2.values(), ticket_list):
    df2.loc[condition, 'Ticket'] = value


honorifics = ['Miss.','Ms.','Mrs.','Master.',
              'Mr.','Rev.','Dr.','Mme.','Don.',
              'Major.','Jonkheer.','Col.','Mlle.',
              'Countess.','Lady.','Sir.','Capt.']


for i in honorifics:
    filtered_rows = df2[df2['Name'].str.contains(i, case=False, na=False)]
    age_mean = round(filtered_rows.loc[:, 'Age'].mean(), 4)
    df2.loc[(df2['Name'].str.contains(i)) & (df2['Age'].isna()), 'Age'] = age_mean


test_embark = df2[(df2['Pclass']==3) & (df2['Sex']=='male') & (df2['Embarked']=='S')]
test_embark_mean = test_embark.loc[:, 'Fare'].mean()
df2.fillna({'Fare': round(test_embark_mean, 4)}, inplace=True)


df2_copy2 = df2.copy()


# drop unneeded cols or cols with many null vals
for i in ['PassengerId','Name','Cabin']:
    if i in df2.columns:
        df2 = df2.drop(i, axis=1)


# one-hot-code non-binary values
cat_variables = ['Pclass',
                 'Sex',
                 'Embarked']


df2 = pd.get_dummies(data=df2,
                         prefix=cat_variables,
                         columns=cat_variables,
                         dtype=int)


PassengerId = df2_copy2['PassengerId']



#-------------------  Predict Test Data  -----------------#



# normalize data
X2 = df2.iloc[:,:]
X2 = preprocessing.StandardScaler().fit(X2).transform(X2)


predictions2 = model.predict(X2)
print(predictions2)


# move 'Survived' column to end
df2.insert(0,'Survived',predictions2)


# create new dataframe of only ids and predictions
df = pd.DataFrame({
    'PassengerId':PassengerId,
    'Survived': df2['Survived']
                  })


# write to csv file
df.to_csv('Titanic_Prediction.csv',index=False)