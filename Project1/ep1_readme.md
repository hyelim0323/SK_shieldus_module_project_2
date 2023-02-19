# Episode_1_blending_model

## data load
```
# kaggle data load
train = pd.read_csv('/content/drive/MyDrive/cloud_share/머신러닝/2차팀플_data/train_module.csv', index_col='id')
test = pd.read_csv('/content/drive/MyDrive/cloud_share/머신러닝/2차팀플_data/test_module.csv', index_col='id')
submission = pd.read_csv('/content/drive/MyDrive/cloud_share/머신러닝/2차팀플_data/sample_submission_module.csv')
```
### 추가 데이터 로드
```
### 추가 데이터 로드
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
original_df = pd.DataFrame(housing.data, columns=housing.feature_names)
original_df[housing.target_names[0]] = housing.target
```
- 원본 train data 37137개 + 추가 train data 20640개 총 57777개의 train data 사용
- test data 24759개

--------------------------------------------------------
![feature_ex](https://user-images.githubusercontent.com/39111185/218971428-e54d73a0-6083-41e2-9039-8976b363fa53.png)
- 독립변수 8개, 종속변수 1개

--------------------------------------------------------
### feature 설명
![feature](https://user-images.githubusercontent.com/39111185/218971755-fdcc7c47-78e9-4463-8040-a2ebfcee7d97.png)

--------------------------------------------------------

## EDA
### library load
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from umap import UMAP

%matplotlib inline
```

### pandas_profiling_report
```
from pandas_profiling import ProfileReport
profile = ProfileReport(train)
profile.to_notebook_iframe()
```
![Independent1](https://user-images.githubusercontent.com/39111185/218974608-8f1c0d96-0be5-48e2-ade8-9ea6c32c3c16.png)
----------------------------------------------------------------------
![Independent2](https://user-images.githubusercontent.com/39111185/218975123-9e9b2c28-67f5-4631-9933-93e8bcd4109d.png)
----------------------------------------------------------------------
![Independent3](https://user-images.githubusercontent.com/39111185/218975130-e14185b2-e413-4805-823c-6147e95fce0b.png)
----------------------------------------------------------------------
![Independent4](https://user-images.githubusercontent.com/39111185/218975148-d42af656-c567-4fe6-942c-e3fa0206a2e2.png)
----------------------------------------------------------------------
![Independent5](https://user-images.githubusercontent.com/39111185/218975159-071e2cc1-332b-43f7-8227-6b68db077a7a.png)
----------------------------------------------------------------------
![Independent6](https://user-images.githubusercontent.com/39111185/218975172-a2901c6d-9f8b-49de-8079-a04a0074499b.png)
----------------------------------------------------------------------
![Independent7](https://user-images.githubusercontent.com/39111185/218975178-4d3a9b43-5cc5-4be7-b430-0ff89ad26529.png)
----------------------------------------------------------------------
![Independent8](https://user-images.githubusercontent.com/39111185/218975214-404ffc7e-a76a-4f42-ba06-9db534d7fa8b.png)
----------------------------------------------------------------------
![Dependent](https://user-images.githubusercontent.com/39111185/218975770-83c72aeb-a444-43f5-8354-2a1646ba723b.png)
----------------------------------------------------------------------
#### Latitude & Longitude
![KakaoTalk_20230213_093010190](https://user-images.githubusercontent.com/39111185/218976257-dc48f63a-041d-40e5-9725-a5fad6f574ca.png)
```
# 위경도별 타겟값 시각화
plt.figure(figsize=(10, 10))
sns.scatterplot(data=train, x='Longitude', y='Latitude', alpha=0.5, hue='MedHouseVal', palette='RdBu_r', size = "MedHouseVal");
```
----------------------------------------------------------------------
#### Heatmap
![heatmap](https://user-images.githubusercontent.com/39111185/218976287-9f58e620-4554-4568-8e7a-71954b603750.png)
----------------------------------------------------------------------
#### correlation
![correlation값](https://user-images.githubusercontent.com/39111185/218976302-86654479-b493-4cf4-a11d-ebc54d96d82b.png)
----------------------------------------------------------------------

## Feature Engineering
### 파생변수 추가
```
# 대회의 내부 데이터인지 sklearn의 원본데이터(외부데이터)인지 구분하는 컬럼 추가
train['internal'] = 1
test['internal'] = 1
original_df['internal'] = 0
# 내부 데이터와 추가 데이터 결합
train = pd.concat([train, original_df], axis=0, ignore_index=True)
```
#### 주택밀집도
```
df['density'] = df['Population'] / df['AveOccup']
```
#### 가구원수 한명 당 평균 방수
```
df['AveRooms_per_Occup'] = df['AveRooms'] / df['AveOccup']
```
----------------------------------------------------------------------
### 위도&경도 파생변수 생성

#### r과 theta값 추가
```
train['r'] = np.sqrt(train['Latitude']**2 + train['Longitude']**2)
train['theta'] = np.arctan2(train['Latitude'], train['Longitude'])

test['r'] = np.sqrt(test['Latitude']**2 + test['Longitude']**2)
test['theta'] = np.arctan2(test['Latitude'], test['Longitude'])
#전처리를 위해 train&test 결합
df = pd.concat([train, test], axis=0, ignore_index=True)
```
#### 카운티
```
# 카운티 데이터 로드
county_gdf = gpd.read_file('/content/drive/MyDrive/cloud_share/머신러닝/자료/us-county-boundaries.geojson')
california_county_gdf = county_gdf[county_gdf['statefp']=='06'].reset_index(drop=True)

# train, test를 공간정보를 포함하는 GeoDataFrame 형식으로 변환
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
gdf = gdf.set_crs(epsg=4326)

# sjoin을 통해 카운티 부여
df = gdf.sjoin(california_county_gdf, how='left')
cols = california_county_gdf.columns.to_list()
cols.remove('geometry')
cols.remove('name')
cols.append('index_right')
df = df.drop(cols, axis=1)

# 카운티별 데이터 개수 집계
df1 = df.pivot_table(index='name', values='MedHouseVal', aggfunc='count')

# 데이터가 100개 이하인 카운티는 'Other'로 치환
other_counties = df1[df1['MedHouseVal'] < 100].index.to_list()
df['name'] = ['Other' if c in other_counties else c for c in df['name']]

# 카운티 정보가 담긴 'name' 컬럼의 컬럼명을 'county'로 변환
df.rename(columns={'name': 'county'}, inplace=True)
```
#### 카운티 시각화
```
import seaborn
palette = seaborn.color_palette('Spectral', 10)
palette

# 타겟값에 따라 색 지정
# 빨간색일수록 타겟값이 크고 파란색일수록 타겟값이 작다
def color_define(val):
  if val < 0.5: return palette.as_hex()[-1]
  elif val < 1: return palette.as_hex()[-2]
  elif val < 1.5: return palette.as_hex()[-3]
  elif val < 2: return palette.as_hex()[-4]
  elif val < 2.5: return palette.as_hex()[-5]
  elif val < 3: return palette.as_hex()[-6]
  elif val < 3.5: return palette.as_hex()[-7]
  elif val < 4: return palette.as_hex()[-8]
  elif val < 4.5: return palette.as_hex()[-9]
  return palette.as_hex()[0]
  
import folium
m = folium.Map(zoom_start=13,
               tiles='http://mt0.google.com/vt/lyrs=m&hl=ko&x={x}&y={y}&z={z}', 
               attr='Google')

# 카운티 시각화
folium.GeoJson(data=california_county_gdf['geometry']).add_to(m)

# 집값 포인트별 시각화
for (lat, lng), block in df[:37137].pivot_table(index=['Latitude', 'Longitude']).iterrows():
  color = color_define(block['MedHouseVal'])
  folium.Circle([lat, lng], radius=100, color=color).add_to(m)

m
```
![county_folium](https://user-images.githubusercontent.com/39111185/219231275-9d5acc73-b2a5-4d1d-bca7-6f2af34bfe97.png)


#### 해안선과 거리
```
# 해안가 데이터 로드
coast_line = gpd.read_file('/content/drive/MyDrive/cloud_share/머신러닝/자료/ne_10m_coastline.shp')

from shapely.geometry import Polygon
# 해안가 데이터을 하나의 도형으로 합집합
coast_lines = coast_line.unary_union

# 데이터별로 해안가와 가장 가까운 거리를 계산
df['distance_to_coast'] = df.distance(coast_lines).values
```
![해안선과거리](https://user-images.githubusercontent.com/39111185/219235962-78e9e2fd-8cee-4f8b-88a4-2bd1e5766c9c.png)

#### 클러스터링
```
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

cluster_df = df[['Latitude', 'Longitude']].copy()
kmeans = KMeans( n_clusters=20, init='k-means++', max_iter=300, random_state=0)
clusters = kmeans.fit_predict(cluster_df)
cluster_df["cluster"] = clusters

x = cluster_df['Latitude']
y = cluster_df['Longitude']
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111)
ax.scatter(x, y, s=40, c=cluster_df["cluster"], marker='o', cmap='Spectral')
ax.set_title("The Plot Of The Clusters")
plt.show()

from haversine import haversine
for idx, cluster_center in enumerate(kmeans.cluster_centers_):
  df[f'distance_to_cluster{idx}'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), cluster_center, unit='mi'), axis=1)
  
#원핫인코딩
df = pd.get_dummies(df)
```
![clustering20](https://user-images.githubusercontent.com/39111185/219237118-b9b45eea-aff8-408a-930b-5dbfd8c86a89.png)

![클러스터링과중심점콕](https://user-images.githubusercontent.com/39111185/219237153-e491bbce-749a-446e-9276-15fb7f05f2ee.png)

#### 주요도시와의 거리

```
Sac = (38.576931, -121.494949)
SF = (37.780080, -122.420160)
SJ = (37.334789, -121.888138)
LA = (34.052235, -118.243683)
SD = (32.715759, -117.163818)

df['dist_Sac'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), Sac, unit='ft'), axis=1)
df['dist_SF'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), SF, unit='ft'), axis=1)
df['dist_SJ'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), SJ, unit='ft'), axis=1)
df['dist_LA'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), LA, unit='ft'), axis=1)
df['dist_SD'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), SD, unit='ft'), axis=1)
df['dist_nearest_city'] = df[['dist_Sac', 'dist_SF', 'dist_SJ', 'dist_LA', 'dist_SD']].min(axis=1)
```
![주요도시와거리](https://user-images.githubusercontent.com/39111185/219235833-a61708a5-42e1-4a64-b0b3-24e4fd6a2300.png)

#### 인코딩 트릭
```
emb_size = 20
precision = 1e6 

latlon = np.expand_dims(df[['Latitude', 'Longitude']].values, axis=-1) 

m = np.exp(np.log(precision) / emb_size) 
angle_freq = m ** np.arange(emb_size) 
angle_freq = angle_freq.reshape(1, 1, emb_size) 

latlon = latlon * angle_freq 
latlon[..., 0::2] = np.cos(latlon[..., 0::2]) 
latlon[..., 1::2] = np.sin(latlon[..., 1::2]) 
latlon = latlon.reshape(-1, 2 * emb_size) 

df['exp_latlon1'] = [lat[0] for lat in latlon]
df['exp_latlon2'] = [lat[1] for lat in latlon]
```
#### PCA & UMAP
```
#PCA
from sklearn.decomposition import PCA
def pca(data):
    '''
    input: dataframe containing Latitude(x) and Longitude(y)
    '''
    coordinates = data[['Latitude','Latitude']].values
    pca_obj = PCA().fit(coordinates)
    pca_x = pca_obj.transform(data[['Latitude', 'Longitude']].values)[:,0]
    pca_y = pca_obj.transform(data[['Latitude', 'Longitude']].values)[:,1]
    return pca_x, pca_y

# train_df['pca_x'], train_df['pca_y'] = pca(train_df)
# test_df['pca_x'], test_df['pca_y'] = pca(test_df)
df['pca_x'], df['pca_y'] = pca(df)

#UMAP
coordinates = df[['Latitude', 'Longitude']].values
umap = UMAP(n_components=2, n_neighbors=50, random_state=228).fit(coordinates)
df['umap_lat'] = umap.transform(coordinates)[:,0]
df['umap_lon'] = umap.transform(coordinates)[:,1]
```
![PCA와UMAP](https://user-images.githubusercontent.com/39111185/219236479-9411ffd7-a5a5-441e-b306-c4d976c05eea.png)

#### rotate encoding
```
def crt_crds(df): 
    df['rot_15_x'] = (np.cos(np.radians(15)) * df['Longitude']) + \
                      (np.sin(np.radians(15)) * df['Latitude'])
    
    df['rot_15_y'] = (np.cos(np.radians(15)) * df['Latitude']) + \
                      (np.sin(np.radians(15)) * df['Longitude'])
    
    df['rot_30_x'] = (np.cos(np.radians(30)) * df['Longitude']) + \
                      (np.sin(np.radians(30)) * df['Latitude'])
    
    df['rot_30_y'] = (np.cos(np.radians(30)) * df['Latitude']) + \
                      (np.sin(np.radians(30)) * df['Longitude'])
    
    df['rot_45_x'] = (np.cos(np.radians(45)) * df['Longitude']) + \
                      (np.sin(np.radians(45)) * df['Latitude'])
    return df

df = crt_crds(df)
```

![rotate](https://user-images.githubusercontent.com/39111185/219236380-19a1d95d-63a8-4b19-b5e0-fdd885c2dbea.png)

----------------------------------------------------------------------
## Modeling

### 평가지표
```
from sklearn.metrics import mean_squared_error
```
![RMSE](https://user-images.githubusercontent.com/39111185/219240897-ae7a7578-8b34-4b68-8586-23dca1f42cfb.png)

### Blending(CatBoost & Lgbm)
```
from sklearn.model_selection import KFold

# regressor의 집합(CatBoost 10개, LightGBM 10개)
regs = []
# 교차검증을 진행하면서 교차 데이터셋에 대한 예측값
y_preds = []
```
#### CatBoost
```
from catboost import CatBoostRegressor

# 교차검증을 10번 수행한다
# 각 교차검증마다 나오는 모델을 저장하여 하나의 알고리즘당 10개의 모델이 생성된다
# 모델의 성능은 10번의 교차검증 점수를 평균내어 평가한다

kfolds = KFold(n_splits=10, random_state=1, shuffle=True)
for train_idx, val_idx in kfolds.split(train):
  X_train = X.iloc[train_idx]
  X_val = X.iloc[val_idx]
  y_train = y.iloc[train_idx]
  y_val = y.iloc[val_idx]

  cat_reg = CatBoostRegressor(**cat_params)
  cat_reg.fit(X_train, y_train,
              early_stopping_rounds=50,
              eval_set=[(X_train, y_train), (X_val, y_val)])
  
  y_pred = cat_reg.predict(X_val)

  y_preds.append(y_pred)
  regs.append(cat_reg)
```
```
cat_rmse = []
for reg in regs[-10:]:
  cat_rmse.append(reg.get_best_score()['validation_1']['RMSE'])
print(f'CatBoost RMSE: {np.mean(cat_rmse)}')
```
#### Lgbm
```
lgbm_params= {
    'n_estimators': 20000,
    'first_metric_only': True,
    'max_depth': 9,
    'metric': 'rmse',
    'random_state': 1
}
```
```
from lightgbm.sklearn import LGBMRegressor

kfolds = KFold(n_splits=10, random_state=1, shuffle=True)
for train_idx, val_idx in kfolds.split(train):
  X_train = X.iloc[train_idx]
  X_val = X.iloc[val_idx]
  y_train = y.iloc[train_idx]
  y_val = y.iloc[val_idx]
  
  lgbm_reg = LGBMRegressor(**lgbm_params)
  lgbm_reg.fit(X_train, y_train,
               early_stopping_rounds=100,
               eval_set=[(X_train, y_train), (X_val, y_val)],
               eval_metric='rmse',
               verbose=1000)
  
  y_pred = lgbm_reg.predict(X_val)

  y_preds.append(y_pred)
  regs.append(lgbm_reg)
```
```
lgbm_rmse = []
for reg in regs[-10:]:
  lgbm_rmse.append(reg.best_score_['valid_1']['rmse'])
print(f'lightGBM RMSE: {np.mean(lgbm_rmse)}')
```
```
lgbm_preds = []
for reg in regs[-10:]:
  lgbm_preds.append(reg.predict(X))
```
#### 예측 및 블렌딩
```
# regs에 저장된 모델들을 이용한 예측값을 y_sub 리스트에 저장한다
y_subs = []
for reg in regs[:]:
  y_subs.append(reg.predict(test)
  
# 저장된 모델만큼 생성된 예측값의 평균을 구해 최종 예측값을 계산한다
# XGB 모델까지 총 세개의 모델로 예측하려 했으나 제외하기로 하였다
y_sub1 = np.array(y_subs[:10]).mean(axis=0)
y_sub2 = np.array(y_subs[10:20]).mean(axis=0)
# y_sub3 = np.array(y_subs[-10:]).mean(axis=0)
y_sub = y_sub1 * 0.5 + y_sub2 * 0.5

```
#### submission
```
submission['MedHouseVal'] = y_sub1
submission.to_csv('result/catboost/submission_catboost_v1.csv', index=False)
submission['MedHouseVal'] = y_sub2
submission.to_csv('result/lightgbm/submission_lightgbm_v1.csv', index=False)
submission['MedHouseVal'] = y_sub
submission.to_csv('result/submission_blending_v1.csv', index=False)
```

#### Feature Importance
```
catboost_feat_imp = []
for reg in regs[:10]:
  catboost_feat_imp.append(reg.feature_importances_)
feature_importance = np.array(catboost_feat_imp).mean(axis=0)
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(12, 7))
sns.barplot(x=feature_importance[sorted_idx][::-1][:15], y=np.array(test.columns)[sorted_idx][::-1][:15]);
plt.title('CatBoost Feature Importance');
```
![catboost_feature_importance](https://user-images.githubusercontent.com/39111185/219249324-2cda643a-c0b0-4d94-b9ae-62871a1011d0.png)

```
lgbm_feat_imp = []
for reg in regs[10:20]:
  lgbm_feat_imp.append(reg.feature_importances_)
feature_importance = np.array(lgbm_feat_imp).mean(axis=0)
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(12, 27))
sns.barplot(x=feature_importance[sorted_idx][::-1][:], y=np.array(test.columns)[sorted_idx][::-1][:]);
plt.title('LightGBM Feature Importance');
```
![FeatureImportanceLGbm](https://user-images.githubusercontent.com/39111185/219249362-339aff99-460d-4670-a44c-323d7d7840a3.png)


### Autogluon
```
from autogluon.tabular import TabularDataset, TabularPredictor

dataset = TabularDataset(train.drop(['geometry'], axis=1))

predictor = TabularPredictor(
    label='MedHouseVal', 
    problem_type='regression', 
    eval_metric='rmse'
    ).fit(
        dataset, 
        presets='best_quality',
num_gpus=1)
```
```
predictor.fit_summary(show_plot=True)
```
```
predictor.get_model_names()
predictor.get_model_best()
```
```
y_pred2 = predictor.predict(test, model='WeightedEnsemble_L3')
submission['MedHouseVal'] = y_pred2
submission.to_csv('submission_WeightedEnsemble_L3.csv', index=False)
```
