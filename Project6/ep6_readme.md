# episode6

## library install
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

## data load
```
train = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/train_episode6.csv', index_col='id')
test = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/test_episode6.csv', index_col='id')
submission = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/sample_submission_episode6.csv')
original = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/ParisHousing.csv')

```
![data_feat](https://user-images.githubusercontent.com/39111185/219530267-a857edc0-0b75-4d3a-832b-d9aa046dd4ef.png)
- Independent 16개 Dependent 1개

![개수](https://user-images.githubusercontent.com/39111185/219530527-8b716e3d-7274-4d36-b3f3-29af896eab83.png)

- train data 총 32730개, test data 총 10000개

## EDA
```
train.hist(figsize=(12, 12));
test.hist(figsize=(12, 12));
original.hist(figsize=(12, 12));
```

![train_hist](https://user-images.githubusercontent.com/39111185/219524187-dbda3eae-e4f0-45ff-9024-11894fb36588.png)
![test_hist](https://user-images.githubusercontent.com/39111185/219524191-e45954c4-8598-4385-9b4b-32d03b6bf0d2.png)
![original_hist](https://user-images.githubusercontent.com/39111185/219524196-bf75ae2e-7a79-4fbb-ab04-f17c80788b70.png)



### data concat
```
train = pd.concat([train , original], sort=False)
df = pd.concat([train, test], ignore_index=True)
```

### 이상치 제거
```
df.squareMeters.sort_values(ascending=False)
dict = {6071330 :99999 ,146181:99999 }
df = df.replace({"squareMeters": dict})
```

```
df.floors.sort_values(ascending=False)
dict = {6000 : 100}
df=df.replace({"floors": dict})
```

```
df.made.sort_values(ascending=False)[:10]
dict = { 10000 : 2021 }
df=df.replace({"made": dict})
```

```
df.basement.sort_values(ascending=False)[:10]
dict = { 91992 : 10000 , 91978 : 10000 ,
        89016 : 10000, 26132 : 10000,
        84333 : 10000  , 81851 :10000 }
df=df.replace({"basement": dict})
```

```
df.attic.sort_values(ascending=False)[:20]
dict = { 96381 : 10000, 
         71965 : 10000 , 
         71024 : 10000, 
         71001 : 10000 ,
        38535 : 10000,
         30000 : 10000,
        13779 : 10000
       }
df=df.replace({"attic": dict})
```

```
df.garage.sort_values(ascending=False)[:10]
dict = { 2048 : 1000, 
         9017 : 1000
       }
df=df.replace({"garage": dict})
```

## 전처리후 EDA

![이상치제거이후](https://user-images.githubusercontent.com/39111185/219528431-9506ea09-1b33-4e8e-8a01-918901c5ac8a.png)

## modeling
```
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = df[~df['price'].isna()]
test = df[df['price'].isna()].drop(['price'], axis=1)

X = train.drop(['price'], axis=1)
y = train['price']
```
```
from sklearn.model_selection import KFold
regs = []

def make_kfold(model):
  kfolds = KFold(n_splits=5, random_state=1, shuffle=True)
  for train_idx, val_idx in kfolds.split(train):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model.fit(X_train, y_train, 
              early_stopping_rounds=10,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              eval_metric='rmse',
              verbose=100)
  
    regs.append(model)
    
xgb_params = {
    'max_depth':4, 
    'learning_rate':0.25 , 
    'n_estimators':500, 
    'objective':'reg:squarederror', 
    'booster':'gbtree'
}

from xgboost import XGBRegressor

xgb_reg = XGBRegressor(**xgb_params)
make_kfold(xgb_reg)
```

```
xgb_rmse = []
for reg in regs[-9:]:
  xgb_rmse.append(np.sqrt(test_y, reg.predict(test_X)))
print(f'XGBoost RMSE: {np.mean(xgb_rmse)}')
```

![rmse_val](https://user-images.githubusercontent.com/39111185/219530045-7a4debc5-548f-4fef-b6f8-8db318b1e112.png)

```
y_subs = []
for reg in regs:
  y_subs.append(reg.predict(test))
y_sub = np.mean(y_subs, axis=0)
```

### Feature Importance 
```
sns.barplot(x=reg.feature_importances_[1:], y=train_X.columns[1:]);
```
![feature_importance](https://user-images.githubusercontent.com/39111185/219529879-1f390552-9b17-414e-beb5-0fec4c87b9e9.png)


## submission 제출
```
submission['price'] = reg.predict(test)
submission.to_csv('submission_ep6.csv', index=False)
```
