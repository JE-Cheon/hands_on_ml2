# 2장 머신러닝 프로젝트 처음부터 끝까지

- 부동산 회사에 고용된 대이터 과학자

## 2.0 주요 단계

1. 큰 그림 보기
2. 데이터 구하기
3. 데이터로부터 통찰을 얻기 위해 탐색하고 시각화
4. 머신러닝 알고리즘을 위한 데이터 준비
5. 모델 선택 및 훈련
6. 모델 상세히 조정
7. 솔루션 제시
8. 시스템 론칭 및 모니터, 유지 보수

---

## 2.1 실제 데이터로 작업하기

- 유명 공개 데이터 저장소
  - [UC 얼바인 머신러닝 저장소](http:/archive.ics.uci.edu/ml)
  - [캐글 데이터셋](http://www.kaggle.com/datasets)
  - [아마존 AWS 데이터셋](https://registry.opendata.aws)
- 메타 포털
  - [데이터 포털](http://dataportals.org)
  - [오픈 데이터 모니터](http://opendatamonitor.eu)
  - [퀀들](http://quandl.com)
- 인기 있는 공개 데이터 저장소 나열 페이지
  - [위키백과 머신러닝 데이터셋 목록](https://goo.gl/SJHN2k)
  - [Quora.com](https://homl.info/10)
  - [데이터셋 서브레딧](http://www.reddit.com/r/datasets)



- StatLib 저장소에 있는 캘리포니아 주택 가격 데이터셋 사용

---

## 2.2 큰 그림 보기

- 캘리포니아 인구조사 데이터를 이용한 주택 가격 모델 생성
- 블록그룹마다 인구, 중간소득, 중간 주택 가격 등 내포



### 2.2.1 문제 정의

- 목적 파악: 문제를 어떻게 구성할지, 어떤 알고리즘을 선택할지, 모델 평가에 어떤 성능 지표를 사용할지, 모델 튜닝을 위해 얼마나 노력할지

![image-20210205171016720](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\02_project\image-20210205171016720.png)

- 파이프라인: 데이터 처리 컴포넌트들이 연속되어 있는 것을 파이프라인이라고 함
  - 컴포넌트들은 일반적으로 비동기적으로 동작
  - 각 컴포넌트는 많은 데이터를 추출하여 처리하고 그 결과를 다른 데이터 저장소로 보냄
  - 일정 시간 후 파이프라인의 컴포넌트가 데이터를 추출하여 출력 결과 도출
  - 각 컴포넌트는 완전히 독립적
- 현재 솔루션 파악
- 레이블된 훈련 샘플을 통한 지도학습, 값 예측을 위한 회귀(다중회귀), 일반적인 배치학습



### 2.2.2 성능 측정 지표 선택

- 회귀 문제의 전형적인 성능 지표: 평균 제곱근 오차(RMSE)

![image-20210205171035377](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\02_project\image-20210205171035377.png)

- 경우에 따라 다른 함수 사용
- 이상치가 많을 경우 평균 절대 오차(MAE) 사용

![image-20210205171048602](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\02_project\image-20210205171048602.png)

- 벡터와 타깃값의 벡터 사이의 거리를 재는 방법(norm)
  - RMSE: 유클리디안 norm
  - 절댓값의 합 계산: 맨해튼 norm
  - norm 지수가 클수록 큰 값의 원소에 치우치며, 작은 값은 무시됨. 그래서 RMSE가 MAE보다 조금 더 이상치에 민감.



### 2.2.3 가정 검사

- 가정 나열 및 검사 필수

---

## 2.3 데이터 가져오기

### 2.3.1 작업환경 만들기

- 작업 디렉터리 생성

```bash
$ export ML_PATH="$HOME/ml" #원하는 경로로 바꿔도 됩니다.
$ mkdir -p $ML_PATH
```

- pip 설치 확인

```bash
$ pip3 --version
```

- 최신버전이 아닐 경우 업그레이트 필요

```bash
$ python3 -m pip install --user -U pip
```

- 독립 환경 만들기(virtualenv 설치)

```bash
$ python3 -m pip install --user -U virtualenv
```

```bash
$ cd $ML_PATH
$ vittualenv env
```

```bash
$ cd $ML_PATH
$ source env/bin/activate #리눅스나 맥os에서
$ .\my_env\Scripts\activate #윈도우에서
```

- 비활성화시 `deactivate` 명령 사용
- pip 명령으로 필요한 패키지와 의존성으로 연결된 다른 패키지 모두 설치

```bash
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
```

- virtualenv 사용시 주피터에 커널 등록 및 이름 설정

```bash
$ python3 -m ipykernel install --user --name=python3
```

- 주피터 노트북 실행

```bash
$ jupyter notebook
```

- 주피터에 새로운 파일 생성

---

### 2.3.2 데이터 다운로드

- 데이터 다운로드 하는 함수 작성 시 편리함

```python
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```

- 판다스를 이용한 데이터 읽어들이기

```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

---

### 2.3.3 데이터 구조 훑어보기

- head() 메서드를 통한 데이터 확인

```python
housing = load_housing_data()
housing.head()
```

- 총 10개의 특성을 갖는 데이터
- info() 메서드는 데이터에 대한 간략한 설명

```python
housing.info()
```

- 범주형 자료의 구역 확인

```python
housing["ocean_proximity"].value_counts()
```

- describe(): 숫자형 특성의 요약 정보 도출

```python
housing.describe()
```

- 데이터 형태 검토(히스토그램)

```python
%matplotlib inline  # 주피터 노트북의 매직 명령(백앤드 지정)
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
```



### 2.3.4 테스트 세트 만들기

- 테스트 세트를 만들지 않는다면 데이터 스누핑 편향 발생 가능
- 데이터 스누핑: 일반화 오차를 추정하여 낙관적인 추정을 만들고 기대한 성능이 나오지 않는 경우
- 난수 인덱스 생성을 통한 테스트 세트 고정

```bash
# 노트북의 실행 결과가 동일하도록
np.random.seed(42)
```

```bash
import numpy as np

# 예시로 만든 것입니다. 실전에서는 사이킷런의 train_test_split()를 사용하세요.
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

- 여러 번 계속할 경우 테스트셋의 의미가 사라짐

  - 해결방법

    - 테스트 세트를 저장한 후 다음 실행에서 불러들이는 것
    - 난수의 초깃값 지정
    - 업데이트된 데이터셋 이용시 문제 발생
    - 샘플의 식별자를 사용하여 테스트세트로 보낼지 말지 결정

    ```bash
    from zlib import crc32
    
    def test_set_check(identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
    
    def split_train_test_by_id(data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]
    ```

  - 식별자 컬럼이 없을 경우 행의 인덱스 사용

    ```bash
    housing_with_id = housing.reset_index()   # `index` 열이 추가된 데이터프레임을 반환합니다
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    ```

    - 인덱스를 고유 식별자로 사용시 새 데이터가 데이터셋의 끝에 추가되어야 함
    - 위 데이터에서는 **위도**와 **경도**를 안정적인 식별자로 사용 가능

    ```bash
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    ```

--- 사이킷런의 원리---

---

- 데이터셋을 여러 서브셋으로 나누는 다양한 방법을 제공하는 사이킷런

- `train_test_split`: 난수 초깃값 지정(`random_state` 매개변수), 행의 개수가 같은 여러 개의 데이터셋을 넘겨 같은 인덱스 기반으로 나누는 것 가능

  ```bash
  from sklearn.model_selection import train_test_split	#순수 무작위 샘플링
  
  train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
  ```

- 샘플이 대표성을 가지기 위해 **계층적 샘플링** 활용

  - 계층적 샘플링: 각 집단의 비율을 유지하며 샘플링하는 방법

- 위 자료의 경우 중간소득이 중간 주택 가격을 예측하는 데에 매우 중요함

- 여러 소득 카테고리 특성을 만들어 분류

- 계층을 나누기 위해 `pd.cut()` 함수 사용

  ```bash
  housing["income_cat"] = pd.cut(housing["median_income"],
                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1, 2, 3, 4, 5])
  ```

  ```bash
  housing["income_cat"].hist()
  ```



- 소득 카테고리를 기반으로 계층 샘플링

```bash
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

- 확인

```bash
strat_test_set["income_cat"].value_counts() / len(strat_test_set) #비율
```

- `income_cat`특성 삭제해서 데이터 원상복구

```bash
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

---

## 2.4 데이터 이해를 위한 탐색과 시각화

- 테스트 세트를 떼어놓았는지 확인 후 훈련 세트에 대해서만 탐색

```bash
housing = strat_train_set.copy()
```



### 2.4.1 지리적 데이터 시각화

- 위도와 경도 덕분에 모든 구역을 산점도로 만들어 데이터 시각화 가능

```bash
housing.plot(kind="scatter", x="longitude", y="latitude")
```

- `alpha`옵션을 통해 밀집된 영역 파악 가능

```bash
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

- 베이 에어리어, 로스엔젤레스, 샌디에고(밀접) / 센트럴벨리, 새크라멘토, 프레즈노 (밀집)

  - 두드러진 패턴 파악을 위해 매개변수 조절

- 주택 가격 시각화

  - 원의 반지름 구역의 인구(매개변수 s)
  - 색상은 가격으로 표현(매개변수 c)
  - 미리 정의된 컬러 맵(`jet`)->매개변수 `cmap`

  ```bash
  housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
      s=housing["population"]/100, label="population", figsize=(10,7),
      c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
      sharex=False)
  plt.legend()
  save_fig("housing_prices_scatterplot")
  ```

  - 주택 가격과 인구밀도는 매우 큰 상관관계를 가짐
  - 군집 알고리즘을 사용하여 주요 군집을 찾고 군집 중심까지의 거리를 재는 특성 추가 가능



### 2.4.2 상관관계 조사

- 데이터셋이 너무 크지 않기 때문에 표준 상관계수(피어슨)`corr()` 매서드 사용

  ```bash
  corr_matrix = housing.corr()
  ```

  ```bash
  corr_matrix["median_house_value"].sort_values(ascending=False)
  ```

  - 상관계수의 범위는 -1~1
  - 1에 가까우면 강한 양의 상관관계를 가짐(중간소득)
  - -1에 가까우면 강한 음의 상관관계를 가짐(위도-약함)
  - 0에 가까우면 선형적인 상관관계가 없음

- 숫자형 특성 사이 산점도를 통해 상관관계 파악 가능

  ```bash
  # from pandas.tools.plotting import scatter_matrix # 옛날 버전의 판다스에서는
  from pandas.plotting import scatter_matrix
  
  attributes = ["median_house_value", "median_income", "total_rooms",
                "housing_median_age"]
  scatter_matrix(housing[attributes], figsize=(12, 8))
  save_fig("scatter_matrix_plot")
  ```

  - 그래프 중 가장 선형관계가 뚜렷해 보이는 그래프를 뽑아냄

  ```bash
  housing.plot(kind="scatter", x="median_income", y="median_house_value",
               alpha=0.1)
  plt.axis([0, 16, 0, 550000])
  save_fig("income_vs_house_value_scatterplot")
  ```

  - `median_income`과 `median_house_value`는 강한 상관관계를 가짐
  - 직선 형태로 보이는 부분은 삭제(알고리즘이 잘못 학습할 가능성 존재)



### 2.4.3 특성 조합으로 실험

- 꼬리가 두꺼운 분포는 변형 필요
- 여러가지 특성 조합 시도
- 가구당 방 개수 파악, 가구당 인원 파악

```bash
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

```bash
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```

- 새롭게 만들어진 관계 속에서 상관관계가 높은 변수가 나올 수 있음



## 2.5 머신러닝 알고리즘을 위한 데이터 준비

- 함수를 만들어 자동화 필요
  - 어떤 데이터셋에 대해서도 데이터 변환을 손쉽게 반복 가능
  - 향후 프로젝트에 사용할 수 있는 변환 라이브러리 점진적 구축
  - 실제 시스템에서 알고리즘에 새 데이터를 주입하기 전에 변환시키는 데에 함수 사용 가능
  - 여러 가지 데이터 변환을 쉽게 시도하고 어떤 조합이 가장 좋은지 확인하는데에 편이
- 원래 훈련 세트로 복원

```bash
housing = strat_train_set.drop("median_house_value", axis=1) # 훈련 세트를 위해 레이블 삭제
housing_labels = strat_train_set["median_house_value"].copy()
```



### 2.5.1 데이터 정제

- 누락된 데이터 확인-`total_bedrooms`

  - 누락된 특성 다루는 방법

    - 해당 구역 제거
    - 전체 특성 삭제
    - 다른 값으로 대체(0,평균,중간값 등)

  - 매서드 이용: `dropna()`, `drop()`, `fillna()`

    ```bash
    housing.dropna(subset=["total_bedrooms"])	#옵션1: 해당 데이터 모두 지우기
    housing.drop("total_bedrooms", axis=1)		#옵션2: 해당 열(특성) 지우기
    median = housing["total_bedrooms"].median()	#옵션3: 해당 na값 median으로 대체, 중간값 저장 필수(inplace=True)
    housing["total_bedrooms"].fillna(median, inplace=True)
    ```

  - 사이킷런의 `SimpleImputer`로 누락된 값 다루기

    ```bash
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")	#누락된 수치 중간값으로 대체
    ```

  - 중간값은 수치형 특성에서만 계산, 텍스트 특성인 `ocean_proximaty` 제외

    ```bash
    housing_num = housing.drop("ocean_proximity", axis=1)
    # 다른 방법: housing_num = housing.select_dtypes(include=[np.number])
    ```

  - `imputer` 객체의 `fit()` 메서드를 사용해 훈련 데이터에 적용 가능

    - `imputer`는 각 특성의 중간값을 계산해서 그 결과를 객체의 `statistics_`속성에 저장

      ```bash
      imputer.fit(housing_num)
      ```

    - 새로운 데이터가 들어 올 경우 누락된 값이 있을 수 있기 떄문에 모든 수치형 특성에 `imputer` 적용

      ```bash
      imputer.statistics_		# imputer를 이용한 중앙값 학습
      housing_num.median().values		# 수동으로 계산한 것과 비교
      ```

    - 학습된 `imputer` 객체를 이용해 훈련 세트에서 누락된 값을 학습한 중간값으로 변경 가능

      ```bash
      X = imputer.transform(housing_num)		#변형된 특성들이 들어 있는 넘파이 배열
      ```

    - 다시 판다스 데이터프레임으로 변경 가능

      ```bash
      housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                                index=housing_num.index)
      ```

- 사이킷런의 설계 철학

  - 일관성: 모든 객체가 일관되고 단순한 인터페이스 공유
    - 추정기: 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체 - `imputer`: `fit()`메서드에 의해 수행됨
    - 변환기: 데이터셋을 변환하는 추정기를 변환기라고 함 - `transform()`, `fit_transform()`
    - 예측기: 주어진 데이터셋에 대한 예측을 만드는 추정기 - `LinearRegression`모델, `predict()` 매서드, 테스트 세트의 `score()` 메서드
  - 검사 가능: 모든 추정기ㅡ이 하이퍼파라미터는 공개 인스턴스 변수로 직접 접근할 수 있고(`imputer.strategy`), 모든 추정기의 학습된 모델 파라미터도 접미사로 밑줄을 붙여 공개 인스턴스 변수로 제공(`imputer.statistics_`)
  - 클래스 남용 방지: 넘파이 배열이나 사이파이 희소 행렬로 표현
  - 조합성: 기존의 구성요소를 최대한 재사용
  - 합리적인 기본값: 대부분의 매개변수에 합리적인 기본값 지정



### 2.5.2 텍스트와 범주형 특성 다루기

- 범주형 입력 특성 전처리

```bash
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
```

- 범주형 텍스트를 숫자로 변환(`OrdinalEncoder`클래스 사용)

```bas
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```

- `categories_` 인스턴스 변수를 이용하여 카테고리 목록을 얻을 수 있음

```bash
ordinal_encoder.categories_
```

- ​	머신러닝 알고리즘이 가까이 있는 두 값이 떨어져 있는 두 값보다 더 비슷하다고 생각함(순서형 자료의 경우 상관 없음)
- 0과 1을 사용하여 서로 상반되는 범주를 이진 특성을 만들어 해결(원-핫인코딩, 더미특성)

```bash
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot	#사이파이 희소행렬 도출
```

- 수천 개의 카테고리가 있는 범주형 특성일 경우 효율적

  - 열이 수천개의 개인 행렬로 변환(각 행에 1이 하나고 1이 있는 위치만 저장)

  - 일반적 2차원 배열처럼 사용 가능, 넘파이 배열로 바꾸려면 `toarray()` 메서드 호출

    ```bash
    cat_encoder.categories_	#카테고리 리스트 출력
    ```



### 2.5.3 나만의 변환기

- 자신만의 변환기를 만들어야 하는 경우 존재

- 만든 변환기를 사이킷런 기능과 연동

- `fit()`, `transform()`, `fit_transform()`메서드를 구현한 파이썬 클래스 생성

- 마지막 메서드는 `TransformMixin` 상속 시 자동으로 생성

- `BaseEstimator` 상속시 하이퍼 파라미터 튜닝에 필요한 두 메서드(`get_params()`, `set_params()`) 얻음

  ```bash
  from sklearn.base import BaseEstimator, TransformerMixin
  
  # 열 인덱스
  rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
  
  class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
      def __init__(self, add_bedrooms_per_room=True): # *args 또는 **kargs 없음
          self.add_bedrooms_per_room = add_bedrooms_per_room
      def fit(self, X, y=None):
          return self  # 아무것도 하지 않습니다
      def transform(self, X):
          rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
          population_per_household = X[:, population_ix] / X[:, households_ix]
          if self.add_bedrooms_per_room:
              bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
              return np.c_[X, rooms_per_household, population_per_household,
                           bedrooms_per_room]
          else:
              return np.c_[X, rooms_per_household, population_per_household]
  
  attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
  housing_extra_attribs = attr_adder.transform(housing.to_numpy())
  ```

  - 변환기가 add_bedrooms_per_room 하이퍼 파라미터 하나를 가짐



### 2.5.4 특성 스케일링

- 모든 특성의 범위를 같도록 만들어주는 방법: **min-max 스케일링**, **표준화**
  - min-max 스케일링(정규화): 0~1 범위에 들도록 값을 이동하고 스케일 조정 - `MinMaxScaler` 변환기 제공, `feature_range`매개변수 범위 조정 가능
  - 표준화: 평균을 빼고 표준편차로 나누어 결과 분포의 분산이 1이 되도록 함, 상한과 하한이 없어 문제 발생 가능, 이상치의 영향을 덜 받음 - `StandardScaler` 변환기



### 2.5.5 변환 파이프라인

- 변환 단계가 많아 순서대로 실행해야 함 - `Pipeline`클래스

  ```bash
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  
  num_pipeline = Pipeline([
          ('imputer', SimpleImputer(strategy="median")),
          ('attribs_adder', CombinedAttributesAdder()),
          ('std_scaler', StandardScaler()),
      ])
  
  housing_num_tr = num_pipeline.fit_transform(housing_num)
  ```

  - `Pipeline`: 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력으로 받음. 변환기와 추정기를 모두 사용할 수 있고 그 외에는 모두 변환기여야 함(`fit_transform()`메서드 내포)
  - `fit()` 메서드 호출 시 모든 변환기의 `fit_transform()` 메서드를 순서대로 호출한 후 단계별 출력을 다음 단계 입력으로 전달, 마지막에는 `fit()` 메서드만 호출
  - 파이프라인 객체는 마지막 추정기와 동일한 메서드 제공, 마지막 추정기가 변환기 `StandardScaler`이기 때문에 파이프라인이 데이터에 대해 모든 변환을 순서대로 적용하는 `transform()` 메서드 내포

- 하나의 변환기로 범주형 열과 수치형 열을 같이 다루는 `ColumnTransformer` 추가

  ```bash
  from sklearn.compose import ColumnTransformer
  
  num_attribs = list(housing_num)
  cat_attribs = ["ocean_proximity"]
  
  full_pipeline = ColumnTransformer([
          ("num", num_pipeline, num_attribs),
          ("cat", OneHotEncoder(), cat_attribs),
      ])
  
  housing_prepared = full_pipeline.fit_transform(housing)
  ```

  - `ColumnTransformer`클래스 임포트
  - 수치형 열 이름의 리스트와 범주형 열 이름의 리스트 생성
  - `ColumnTransformer`클래스 객체 생성 - 튜플 리스트 받음
  - 수치형 열은 `num_pipeline`, 범주형 열은 `OneHotEncoder`를 사용하여 변환
  - `ColumeTransformer`를 주택 데이터에 적용



## 2.6 모델 선택과 훈련

### 2.6.1 훈련 세트에서 훈련하고 평가하기

- 선형 회귀 모델

```bash
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

- 적용 예시

  ```bash
  # 훈련 샘플 몇 개를 사용해 전체 파이프라인을 적용해 보겠습니다
  some_data = housing.iloc[:5]
  some_labels = housing_labels.iloc[:5]
  some_data_prepared = full_pipeline.transform(some_data)
  
  print("예측:", lin_reg.predict(some_data_prepared))
  print("레이블:", list(some_labels))
  ```

- `mean_square_error` 함수를 이용한 RMSE 측정

  ```bas
  from sklearn.metrics import mean_squared_error
  
  housing_predictions = lin_reg.predict(housing_prepared)
  lin_mse = mean_squared_error(housing_labels, housing_predictions)
  lin_rmse = np.sqrt(lin_mse)
  lin_rmse
  ```

  - 예측오차가 클 경우 모델이 훈련 데이터에 과소적합된 사례
  - 좋은 예측을 만들 만큼 충분한 정보를 제공하지 못했거나 모델이 충분히 강력하지 못함
  - 더 강력한 모델을 선택하거나 훈련 알고리즘에 더 좋은 특성을 주입하거나 무델의 규제를 감소시켜 과소적합 해결 가능

- 더 복잡한 모델 훈련(`DecisionTreeregressor` 훈련)

  ```bash
  from sklearn.tree import DecisionTreeRegressor
  
  tree_reg = DecisionTreeRegressor(random_state=42)
  tree_reg.fit(housing_prepared, housing_labels)
  ```

  - 훈련 세트 평가

    ```bash
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse
    ```

    - 데이터가 심하게 과대적합되어 오차 발생 안함

### 2.6.2 교차 검증을 사용한 평가

- 검정 트리 모델 평가

- `train_test_split`함수를 사용해 훈련세트를 더 작은 훈련세트와 검증세트로 나눔

- 더 작은 훈련 세트에 모델 훈련

- 검증세트로 모델 평가

- **k-겹 교차 검증**기능 사용: 훈련 세트를 **폴드**라 불리는 10개의 서브셋으로 무작위 분할

- 결정 트리 모델을 10번 훈련 및 평가, 매번 다른 폴드 선택 및 평가, 나머지 9개의 폴드는 훈련에 사용

- 10개의 평가 점수가 담긴 배열이 결과

  ```bash
  from sklearn.model_selection import cross_val_score
  
  scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                           scoring="neg_mean_squared_error", cv=10)
  tree_rmse_scores = np.sqrt(-scores)
  ```

  - 결과(별로 좋지 않음)

    ```bash
    def display_scores(scores):
        print("점수:", scores)
        print("평균:", scores.mean())
        print("표준 편차:", scores.std())
    
    display_scores(tree_rmse_scores)
    ```

- 선형 회귀 모델

  ```bash
  lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                               scoring="neg_mean_squared_error", cv=10)
  lin_rmse_scores = np.sqrt(-lin_scores)
  display_scores(lin_rmse_scores)
  ```

- `RandomForestRegressor` 모델

  ```bash
  from sklearn.ensemble import RandomForestRegressor
  
  forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
  forest_reg.fit(housing_prepared, housing_labels)
  housing_predictions = forest_reg.predict(housing_prepared)
  forest_mse = mean_squared_error(housing_labels, housing_predictions)
  forest_rmse = np.sqrt(forest_mse)
  forest_rmse
  
  from sklearn.model_selection import cross_val_score
  
  forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                  scoring="neg_mean_squared_error", cv=10)
  forest_rmse_scores = np.sqrt(-forest_scores)
  display_scores(forest_rmse_scores
  ```

  - 특성을 무작위로 선택하여 많은 결정 트리를 만들고 그 예측을 평균 내는 방식
  - 여러 다른 모델을 모아서 하나의 모델을 만드는 것 **앙상블 학습**



## 2.7 모델 세부 튜닝

### 2.7.1 그리드 탐색

- 가장 단순한 방법은 하이퍼파라미터 조합을 찾을 때까지 수동으로 조정하는 것

- `GridSearchCV` 사용시 시도해볼만한 값 지정

- `RandomForestRegressor` 에 대한 최적의 하이퍼파라미터 조합 탐색

  ```bash
  from sklearn.model_selection import GridSearchCV
  
  param_grid = [
      # 12(=3×4)개의 하이퍼파라미터 조합을 시도합니다.
      {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
      # bootstrap은 False로 하고 6(=2×3)개의 조합을 시도합니다.
      {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
  
  forest_reg = RandomForestRegressor(random_state=42)
  # 다섯 개의 폴드로 훈련하면 총 (12+6)*5=90번의 훈련이 일어납니다.
  grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                             scoring='neg_mean_squared_error',
                             return_train_score=True)
  grid_search.fit(housing_prepared, housing_labels)
  ```

  - `param_grid` 설정에 따라 사이킷런이 첫 번째 `dict`에 있는 `n_estimators`와 `max_features`하이퍼 파라미터 조합인 **3*4=12**개 평가

  - 두 번째 하이퍼 파라미터 계산

  - 모두 합해서 계산 후 모델 훈련을 통한 최적의 조합을 얻음

    ```bash
    grid_search.best_params_
    ```

- 최적의 추정기에 직접 접근도 가능

  ```bash
  grid_search.best_estimator_
  ```

- 평가 점수 확인 가능

```bash
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params
```

	- 최적의 파라미터 조합 서치



### 2.7.2 랜덤 탐색

- 하이퍼파라미터의 탐색 공간이 커질 경우 `RandomizedSearchCV` 사용
  - 랜덤 탐색을 1000회 반복하도록 실행하면 하이퍼파라미터마다 각기 다른 1000개의 값을 탐색
  - 단순히 반복 횟수를 조절하는 것만으로 하이퍼파라미터 탐색에 투입할 컴퓨팅 자원 제어



### 2.7.3 앙상블 방법

- 최상의 모델 연결
- 모델 그룹이 최상의 단일 모델보다 더 나은 성능을 발휘하는 경우가 있음
- 개개의 모델이 각기 다른 형태의 오차를 만들 때



### 2.7.4 최상의 모델과 오차 분석

- 최상의 모델 분석을 통해 문제에 대한 좋은 통찰을 얻을 수 있음

  ```bash
  feature_importances = grid_search.best_estimator_.feature_importances_
  feature_importances
  ```

- 특성 이름 표시

  ```bash
  extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
  #cat_encoder = cat_pipeline.named_steps["cat_encoder"] # 예전 방식
  cat_encoder = full_pipeline.named_transformers_["cat"]
  cat_one_hot_attribs = list(cat_encoder.categories_[0])
  attributes = num_attribs + extra_attribs + cat_one_hot_attribs
  sorted(zip(feature_importances, attributes), reverse=True)
  ```

- 이 정보를 통해 덜 중요한 특성들 제외 가능



### 2.7.5 테스트 세트로 시스템 평가하기

- 테스트 세트에서 최종 모델 평가

- 테스트 세트에서 예측 변수와 레이블을 얻은 후 `full_pipeline`을 사용하여 데이터 변환

- 테스트세트에서 최종 모델 평가

  ```bash
  final_model = grid_search.best_estimator_
  
  X_test = strat_test_set.drop("median_house_value", axis=1)
  y_test = strat_test_set["median_house_value"].copy()
  
  X_test_prepared = full_pipeline.transform(X_test)
  final_predictions = final_model.predict(X_test_prepared)
  
  final_mse = mean_squared_error(y_test, final_predictions)
  final_rmse = np.sqrt(final_mse)
  ```

- 이 때 일반화 오차의 추정이 론칭을 결정하기에 충분하지 않음

- `scipy.stats.t.interval()`을 사용하여 일반화 오차의 95% 신뢰구간 계산 가능

  ```bash
  from scipy import stats
  
  confidence = 0.95
  squared_errors = (final_predictions - y_test) ** 2
  np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                           loc=squared_errors.mean(),
                           scale=stats.sem(squared_errors)))
  ```

  - 하이퍼파라미터 튜닝을 많이 했다면 교차 검증을 사용하여 측정한 것보다 조금 성능이 낮은 것이 보통
  - 이 때 튜닝x, 향상된 성능이 새로운 데이터에 일반화되기 어려움



## 2.8 론칭, 모니터링, 시스템 유지 보수

- 제품 시스템에 적용하기 위한 준비 필요

- 모델을 상용 환경에 배포 가능

  - 전체 전처리 파이프라인과 예측 파이프라인이 포함된 훈련된 사이킷런 모델을 저장
  - `predict()` 메서드를 호출하여 예측을 만듦

  ![image-20210205171136182](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\02_project\image-20210205171136182.png)

- 구글 클라우드 AI 플랫폼과 같은 클라우드에 배포
  - 모델 저장 및 구글 클라우드 스토리지에 업로드
  - 플랫폼으로 이동하여 새로운 모델 버전을 만들고 GCS 파일을 지정
- 배포 후 일정 간격으로 시스템의 실시간 성능 체크, 성능이 떨어졌을 때 알람을 통지할 수 있는 모니터링 코드 작성 필요
  - 하위 시스템의 지표로 모델 성능 추정
  - 모델이 분류한 전체 사진 중 한 샘플을 평가하는 사람에게 보냄
- 모델이 실패했을 떄 무엇을 할지 정의하고 어떻게 대비할지 관련 프로세스 대비
- 변화하는 데이터에 따른 정기적인 훈련 필요

- 자동화 가능 부분
  - 정기적으로 새로운 데이터를 수집하고 레이블을 닮
  - 모델을 훈련하고 하이퍼파라미터를 자동으로 세부 튜닝하는 스크립트 작성
  - 업데이트된 테스트 세트에서 새로운 모델과 이 전 모델을 평가하는 스크립트 작성
- 모델의 입력 데이터 품질 평가(알람)
- 만들 모델 백업 필요



