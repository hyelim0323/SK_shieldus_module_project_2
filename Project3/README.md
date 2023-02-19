# Sk Shieldus Rookies 머신러닝 미니 프로젝트 (5조)

## Kaggle Season 3, Episode 3
- Tabular Classification with an Employee Attrition Dataset
- https://www.kaggle.com/competitions/playground-series-s3e3

- 실행 환경
  - Google Colab
  - Python 3.8.10
  - Sklearn 1.2.1
  - CatBoost 1.1.1
 
- Blending 시, Colab 기본 환경 사용 (Sklearn 업데이트 X)

---

# 목표
- **0.897 넘기기 (10%(66등) 안에 들기)**

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

# 다운받아서 불러옴
ibm = pd.read_csv(data_path + 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
```

- 모델 성능 향상 데이터 추가
- https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- train.shape : (1677, 34)
- test.shape : (1119, 33)

## 불필요 데이터 삭제
```
train.drop(columns=['Over18', 'EmployeeCount', 'StandardHours'], inplace=True, errors='ignore')
test.drop(columns=['Over18', 'EmployeeCount', 'StandardHours'], inplace=True, errors='ignore')
```

### train 데이터

||feature_name|	type|	결측값수|	고유값수|	샘플값 0|	샘플값 1|	샘플값 2|
|---|---|---|---|---|---|---|---|
|0|	Age|	int64|	0|	43|	36|	35|	32|
|1|	BusinessTravel|	object|	0|	3|	Travel_Frequently|	Travel_Rarely|	Travel_Rarely|
|2|	DailyRate|	int64|	0|	901|	599|	921|	718|
|3|	Department|	object|	0|	3|	Research & Development|	Sales|	Sales|
|4|	DistanceFromHome|	int64|	0|	29|	24|	8|	26|
|5|	Education|	int64|	0|	6|	3|	3|	3|
|6|	EducationField|	object|	0|	6|	Medical|	Other|	Marketing|
|7|	EnvironmentSatisfaction|	int64|	0|	4|	4|	1|	3|
|8|	Gender|	object|	0|	2|	Male|	Male|	Male|
|9|	HourlyRate|	int64|	0|	71|	42|	46|	80|
|10|	JobInvolvement|	int64|	0|	4|	3|	3|	3|
|11|	JobLevel|	int64|	0|	6|	1|	1|	2|
|12|	JobRole|	object|	0|	9|	Laboratory| Technician|	Sales| Representative|	Sales| Executive|
|13|	JobSatisfaction|	int64|	0|	4|	4|	1|	4|
|14|	MaritalStatus|	object|	0|	3|	Married|	Married|	Divorced|
|15|	MonthlyIncome|	int64|	0|	1383|	2596|	2899|	4627|
|16|	MonthlyRate|	int64|	0|	1447|	5099|	10778|	16495|
|17|	NumCompaniesWorked|	int64|	0|	10|	1|	1|	0|
|18|	OverTime|	object|	0|	2|	Yes|	No|	No|
|19|	PercentSalaryHike|	int64|	0|	15|	13|	17|	17|
|20|	PerformanceRating|	int64|	0|	2|	3|	3|	3|
|21|	RelationshipSatisfaction|	int64|	0|	4|	2|	4|	4|
|22|	StockOptionLevel|	int64|	0|	4|	1|	1|	2|
|23|	TotalWorkingYears|	int64|	0|	41|	10|	4|	4|
|24|	TrainingTimesLastYear|	int64|	0|	7|	2|	3|	3|
|25|	WorkLifeBalance|	int64|	0|	4|	3|	3|	3|
|26|	YearsAtCompany|	int64|	0|	38|	10|	4|	3|
|27|	YearsInCurrentRole|	int64|	0|	19|	0|	2|	2|
|28|	YearsSinceLastPromotion|	int64|	0|	16|	7|	0|	1|
|29|	YearsWithCurrManager|	int64|	0|	18|	8|	3|	2|
|30|	Attrition|	int64|	0|	2|	0|	0|	0|


  - train.shape : (1677, 31)
  - test.shape : (1119, 30)

- 시각화 결과는 .ipynb 파일 참고

## Feature Engineering
### 추가 데이터 전처리

```
# ibm 피쳐 개수가 많아 보임 (전처리 수행)
ibm['Attrition'] = (ibm['Attrition'] == 'Yes').astype(np.int64)

ibm.drop(columns="EmployeeNumber", inplace=True)

ibm = ibm[list(train.columns)]

# train + ibm 
train = pd.concat([train, ibm]).reset_index(drop=True)
```
  - train.shape : (3147, 31)
  - test.shape : (1119, 30)

### 불필요한 데이터  삭제 및 순서형 데이터 정렬

```

# 이상치 제거
train.drop(train[train['JobLevel']==7].index.tolist() , inplace=True, errors='ignore')
# DailyRate와 Education에 이상치 제거 (train만, test에는 해당 이상치 없음)
train.drop([527, 1398], inplace=True)

# 순서형 데이터 정렬
ord_1 = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
# 순서생성
ord_1_dtype = CategoricalDtype(ord_1, True)
# 순서 적용 -> Dtype 적용
train['BusinessTravel'] = train['BusinessTravel'].astype(ord_1_dtype)
```
## Data Encoding

- 원-핫 인코딩 X => CatBoost는 범주형 데이터 인코딩을 자체적으로 해준다.

### 이진형 데이터 인코딩

- 이진형 데이터 0과 1로 변환

```
# train data
train[bin_datas[0]] = train[bin_datas[0]].map({'Yes':1, 'No':0})
train[bin_datas[1]] = train[bin_datas[1]].map({'Male':1, 'Female':0})
train[bin_datas[2]] = train[bin_datas[2]].map({3:1, 4:0})

# test data
test[bin_datas[0]] = test[bin_datas[0]].map({'Yes':1, 'No':0})
test[bin_datas[1]] = test[bin_datas[1]].map({'Male':1, 'Female':0})
test[bin_datas[2]] = test[bin_datas[2]].map({3:1, 4:0})
```

### 명목형 데이터 인코딩

- 'BusinessTravel' 피쳐는 순서대로 지정을 해도 될 것 같다.
- 'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2 순서대로 인코딩
- 실제로 여행을 많이 간 순서대로 나열했더니, 결과 값이 올랐다.

```
ord_1_dict = { 'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2 }

train['BusinessTravel'] = train['BusinessTravel'].map( ord_1_dict )
test['BusinessTravel'] = test['BusinessTravel'].map( ord_1_dict )

# category => int
train['BusinessTravel'] = train['BusinessTravel'].astype(int)

test['BusinessTravel'] = test['BusinessTravel'].astype(int)
```

## Data Scaling

- 순서형, 수치형 -> StandardScaler (평균 0, 표준편차 1이 되도록 모든 값을 조정하여 변환)
- 학습 시 메모리를 적게 사용하기 위해 스케일링 진행
- 모델에 따라 스케일링이 결과에 크게 영향을 미치지 않을수도 있다.

```
scaler = StandardScaler().fit(train[ord_datas])
train[ord_datas] = scaler.transform(train[ord_datas])
test[ord_datas] = scaler.transform(test[ord_datas])

scaler = StandardScaler().fit(train[num_datas])
train[num_datas] = scaler.transform(train[num_datas])
test[num_datas] = scaler.transform(test[num_datas])
```

## Modeling

- 최종 CatBoost 모델 사용

```
X = train.drop('Attrition', axis=1)
y = train['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0, shuffle=True)

pool_train = Pool(X_train,y_train,cat_features=cat_features)
pool_test = Pool(X_test,y_test,cat_features=cat_features)

clf_pram = {
    'learning_rate':0.1835,
    'random_seed' : 7,
    'iterations':200,
    'loss_function':'Logloss',
    'depth':2,
    'subsample':0.89,
    'verbose':100,
    'bootstrap_type' : 'Bernoulli', 
    'l2_leaf_reg' : 7
}
cat = CatBoostClassifier(**clf_pram)
cat.fit(pool_train, eval_set=pool_test, use_best_model=True)
```

### 결과 
- Private Score : 0.90037 => 12등
- Public score : 0.94195

## 모델 성능 향상을 위한 Blending
- https://www.kaggle.com/code/bcruise/starting-strong-xgboost-lightgbm-catboost?scriptVersionId=116642571
- 1위(XGBoost + LightGBM + CatBoost) + 생성한 모델(CatBoost)
- 1위의 모델(XGBoost, LightGBM, CatBoost)와 우리팀의 CatBoost모델이 낸 예측값들을 블렌딩
- 총 네개의 모델(XGBoost, LightGBM, CatBoost2)을 블렌딩한 예측값 `score : 0.90407` (1등)
```
final_preds = np.column_stack([xgb_preds, xgb_preds,
                               cat_preds, local_best['Attrition'].values]).mean(axis=1)
```
# 최종 결과

Private Score : 0.90407 (1등)
