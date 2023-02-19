# Sk Shieldus Rookies 머신러닝 미니 프로젝트 (5조)

## Kaggle Season 3, Episode 2
- Tabular Classification with a Stroke Prediction Dataset
- https://www.kaggle.com/competitions/playground-series-s3e2

- 실행 환경
  - Google Colab
  - Python 3.8.10
  - Sklearn 1.2.1
  - XGBoost 0.90

---

# 목표
- **0.89714 넘기기 (10%(77등) 안에 들기)**

---


# 데이터 준비

```
import numpy as np
import pandas as pd

# 데이터 경로
data_path = '/content/'

train = pd.read_csv(data_path + 'train.csv',index_col='id')
test = pd.read_csv(data_path + 'test.csv',index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv',index_col='id')

original_data = pd.read_csv(data_path + 'healthcare-dataset-stroke-data.csv',index_col='id')
```

- 모델 성능 향상 데이터 추가
- https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- train.shape : (15304, 11)
- test.shape : (10204, 10)

### train 데이터

|index|	feature_name|	type|	결측값수|	고유값수|	샘플값 0|	샘플값 1|	샘플값 2|
|---|---|---|---|---|---|---|---|
|0|	gender|	object|	0|	3|	Male|	Male|	Female|
|1|	age|	float64|	0|	106|	28.0|	33.0|	42.0|
|2|	hypertension|	int64|	0|	2|	0|	0|	0|
|3|	heart_disease|	int64|	0|	2|	0|	0|	0|
|4|	ever_married|	object|	0|	2|	Yes|	Yes|	Yes|
|5|	work_type|	object|	0|	5|	Private|	Private|	Private|
|6|	Residence_type|	object|	0|	2|	Urban|	Rural|	Rural|
|7|	avg_glucose_level|	float64|	0|	3740|	79.53|	78.44|	103.0|
|8|	bmi|	float64|	0|	407|	31.1|	23.9|	40.3|
|9|	smoking_status|	object|	0|	4|	never| smoked|	formerly| smoked|	Unknown|
|10|	stroke|	int64|	0|	2|	0|	0|	0|

- 시각화 결과는 .ipynb 파일 참고

## Feature Engineering
### 추가 데이터 전처리
- original[bmi] Null 값 채우기

```
# original bmi 평균값으로 채움
original_data["bmi"].fillna(original_data["bmi"].mean(),inplace=True)
original_data = original_data[list(train.columns)]

# 훈련 데이터셋 증강
train = pd.concat([train, original_data]).reset_index(drop=True)
```
- train.shape : (20414, 11)
- test.shape : (10204, 10)

## Data Encoding

### 불필요 데이터 처리 및 이진형 데이터 인코딩

- 이진형 데이터 0과 1로 변환 ('gender'에서 'Other' 값 처리)

```
bin_enc_feats = ['ever_married', 'Residence_type', 'gender']

train[bin_enc_feats[2]].mode()[0], test[bin_enc_feats[2]].mode()[0]

# Female이 최빈값으로 나온다.
```

```
# 'Other' 최빈값으로 처리
# 기타 이진형 데이터 0과 1로 처리
train[bin_enc_feats[0]] = train[bin_enc_feats[0]].map({'Yes':1, 'No':0})
train[bin_enc_feats[1]] = train[bin_enc_feats[1]].map({'Urban':1, 'Rural':0})
train[bin_enc_feats[2]] = train[bin_enc_feats[2]].map({'Male':1, 'Female':0, 'Other':0})

test[bin_enc_feats[0]] = test[bin_enc_feats[0]].map({'Yes':1, 'No':0})
test[bin_enc_feats[1]] = test[bin_enc_feats[1]].map({'Urban':1, 'Rural':0})
test[bin_enc_feats[2]] = test[bin_enc_feats[2]].map({'Male':1, 'Female':0, 'Other':0})
```

### 명목형 데이터 인코딩

```
nom_datas = ['work_type', 'smoking_status']

# 피쳐가 6개 이하이기 때문에 원-핫 인코딩 처리
from sklearn.preprocessing import OneHotEncoder

enc_nom_train = OneHotEncoder().fit_transform( np.array( train[nom_datas] ) )
enc_nom_test = OneHotEncoder().fit_transform( np.array( test[nom_datas] ) )
```
- enc_nom_train.shape : (20414, 9)
- enc_nom_test.shape : (10204, 9)


## Data Scaling

- 수치형 -> MinMaxScaler (최댓값은 1로, 최솟값은 0으로 데이터의 범위를 조정)
- 데이터와 모델에 맞게 스케일링해주는게 좋다.
- 실제로 결과 값에 영향을 주는 스케일링이었다.
- 때에 따라서는 크게 영향이 없을 수도 있다.

```
num_datas = ['age', 'avg_glucose_level', 'bmi' ]

enc_num_train = MinMaxScaler().fit_transform(train[num_datas])
enc_num_test = MinMaxScaler().fit_transform(test[num_datas])
```

## 데이터 병합
### df의 데이터와 CSR 데이터가 합병 => CSR로 합병
- train
    - train[bin_features].shape : (20414, 5)
    - enc_nom_train.shape : (20414, 9)
    - enc_num_train.shape : (20414, 3)

```
final_train_data_csr = sparse.hstack( [
    train[bin_features],        # 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'gender' 
    enc_nom_train,              # 'work_type', 'smoking_status' 
    enc_num_train               # 'age', 'avg_glucose_level', 'bmi' 
], format='csr')
```

- test
    - test[bin_features].shape : (10204, 5)
    - enc_nom_test.shape : (10204, 9)
    - enc_num_test.shape : (10204, 3)
```
final_test_data_csr = sparse.hstack( [
    test[bin_features],        # 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'gender' 
    enc_nom_test,              # 'work_type', 'smoking_status' 
    enc_num_test               # 'age', 'avg_glucose_level', 'bmi' 
], format='csr')
```


## Modeling

- 최종 XGBoost 모델 사용

```
X = final_train_data_csr
y = train['stroke']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.14985, random_state=0, shuffle=True)

eval_set = [(X_test, y_test)]

clf_pram = {
    'n_estimators':200,
    'learning_rate' : 0.1199,
    'max_depth':2,
    'subsample':0.88,
    'n_jobs':-1,
    'eval_metric':'logloss',
    'reg_lambda': 15.02,
    'seed': 9,
    'colsample_bytree': 0.9,
    'min_child_weight': 7.7
}

xgb = XGBClassifier(**clf_pram)

xgb.fit( X_train, y_train , verbose=100, early_stopping_rounds=30, eval_set=eval_set)
```

# 최종 결과

- 최종 사용 모델 : XGBoost
- 최종 결과 (3등, 금메달)
  - Private Score : 0.90059
  - Public Score : 0.86477

