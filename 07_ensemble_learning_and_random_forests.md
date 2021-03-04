# 7장 앙상블 학습과 랜덤 포레스트

- 일련의 예측기로부터 예측을 수집하면 가장 좋은 모델 하나보다 더 좋은 예측을 얻을 수 있음 (**앙상블**)
  - ex) 훈련 세트로부터 무작위로 각기 다른 서브셋을 만들어 일련의 결정트리 분류기를 훈련시키는 것 가능
- **랜덤 포레스트**: 결정트리의 앙상블
- 프로젝트의 마지막에는 흔히 앙상블 방법을 사용하여 여러 괜찮은 예측기를 연결하여 더 좋은 예측기를 만듦
- 앙상블 방법에는 **배깅, 부스팅, 스태킹** 등이 있음



## 7.1 투표 기반 분류기

- 정확도가 80%인 분류기 여러 개를 훈련시켰다고 가정

  ![image-20210227014748629](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210227014748629.png)

- 더 좋은 분류기를 만드는 간단한 방법은 각 분류기의 예측을 모아 가장 많이 선택된 클래스 예측

- **직접 투표** 분류기

  ![image-20210227015225290](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210227015225290.png)

  - 위 앙상블 분류기는 개별 분류기 중 가장 뛰어난 것보다 정확도가 높을 수 있음
  - **약한 학습기**들이 모여 앙상블을 하면 **강한 학습기**가 됨
  
- 같은 사건을 많은 횟수로 반복했을 경우 **큰 수의 법칙**에 의해 수학적 확률에 가까워짐

  ![image-20210301222954190](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210301222954190.png)

- 51%의 정확도를 가진 1000개의 분류기로 앙상블 모델을 구축한다고 가정

  - 가장 많은 클래스를 예측으로 삼으면 75%의 정확도를 기대
  - 모든 분류기가 완벽하게 독립적이지 않을 경우 같은 종류의 오차를 만들기 쉽기 때문에 정확도가 낮아짐

- 사이킷런의 투표 기반 분류기(`VotingClassifier`) 생성 및 훈련

  ```bash
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import VotingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  
  log_clf = LogisticRegression(solver="lbfgs", random_state=42)
  rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
  svm_clf = SVC(gamma="scale", random_state=42)
  
  voting_clf = VotingClassifier(
      estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
      voting='hard')
  ```

  - 각 분류기의 테스트셋 정확도 확인

    ```bash
    from sklearn.metrics import accuracy_score
    
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    ```

    - 투표 기반 분류기가 다른 개변 분류기보다 성능이 조금 더 높음

- 모든 분류기가 클래스의 확률을 예측할 수 있을 경우, 개별 분류기의 예측을 평균 내어 확률이 가정 높은 클래스 예측 가능(**간접투표**)



## 7.2 배깅과 페이스팅

- 다양한 분류기를 만드는 한 가지 방법은 각기 다른 훈련 알고리즘을 사용하는 것

- 같은 알고리즘을 사용하고 훈련 세트의 서브셋을 무작위로 구성하여 분류기를 각기 다르게 학습시키는 것

- **배깅**: 훈련 세트에서 중복을 허용하여 샘플링 하는 방식

- **페이스팅**: 중복을 허용하지 않고 샘플링하는 방식

- 둘 중 배깅만 한 예측기를 위해 같은 훈련 샘플을 여러 번 샘플링 가능

  ![image-20210301225559320](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210301225559320.png)

  - 모든 예측기가 훈련을 마치면 앙상블은 모든 예측기의 예측을 모아 새로운 샘플에 대한 예측을 만듦
  - 분류 => **통계적 최빈값**, 회귀 =>평균
  - 수집함수 통과시 편향과 분산이 모두 감소
  - 앙상블 결과 => 원본 데이터셋으로 하나의 예측기를 훈련시킬 때와 비교하여 편향은 비슷하지만 분산은 줄어듦

- 예측기는 모두 동시에 병렬로 학습, 예측 또한 병렬로 수행



### 7.2.1 사이킷런의 배깅과 페이스팅

- `BaggingClassifier`를 통해 배깅과 페이스팅 수행 가능

- 결정 트리 분류기 500개의 앙상블 훈련 코드

  ```bash
  from sklearn.ensemble import BaggingClassifier
  from sklearn.tree import DecisionTreeClassifier
  
  bag_clf = BaggingClassifier(
      DecisionTreeClassifier(random_state=42), n_estimators=500,
      max_samples=100, bootstrap=True, random_state=42)
  bag_clf.fit(X_train, y_train)
  y_pred = bag_clf.predict(X_test)
  ```

  - 페이스팅 사용 시 `bootstrap=False`로 지정

  - `n_jobs` 매개변수는 사이킷런이 훈련과 예측에 사용할 CPU 코어 수 지정

    ![image-20210302010211372](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302010211372.png)

  - 단일 결정 트리의 결정 경계와 500개의 트리를 사용한 배깅 앙상블의 결정경계 비교

  - moons 데이터셋 훈련

  - 배깅 앙상블이 더 일반화가 잘 됨

  - 비슷한 편향에서 더 작은 분산을 만듦

- 부트스트래핑은 각 예측기가 학습하는 서브셋에 다양성을 증가시킴

  - 배깅이 페이스팅보다 편향이 조금 더 높음
  - 다양성을 추가한다는 것 =>  예측기들의 상관관계를 줄이기 때문에 앙상블의 분산 감소시킴



### 7.2.2 oob 평가

- `BaggingClassifier`: 기본값으로 중복을 허용하여 훈련세트의 크기만큼인 *m*개 샘플을 선택 (평균 63%정도 샘플링)

- 훈련 샘플의 나머지 37%를 oob샘플이라고 부름

- 예측기가 훈련되는 동안 oob샘플을 사용하지 않기 떄문에 별도의 검증 세트를 사용하지 않고 oob샘플을 사용해 평가 가능

- 앙상블 평가는 각 예측기의 oob평가를 평균하여 얻음

- `BaggingClassifier`에 `oon_score=True`로 지정하면 훈련이 끝난 후 자동으로 oob평가 수행

  ```bash
  bag_clf = BaggingClassifier(
      DecisionTreeClassifier(random_state=42), n_estimators=500,
      bootstrap=True, oob_score=True, random_state=40)
  bag_clf.fit(X_train, y_train)
  bag_clf.oob_score_
  ```

  - 평가 점수는 `oob_score_` 변수에 저장

    ```bash
    from sklearn.metrics import accuracy_score
    y_pred = bag_clf.predict(X_test)
    accuracy_score(y_test, y_pred)		# 확인작업
    ```

    - oob샘플에 대한 결정 함수 값

    ```bash
    bag_clf.oob_decision_function_
    ```



## 7.3 랜덤 패치와 랜덤 서브스페이스

- `BaggingClassifier`는 특성 샘플링 지원
- 샘플링은 `max_features`, `bootstrap_features` 매개변수로 조절
- 샘플이 아닌 특성에 대한 샘플링
  - 무작위로 선택한 입력 특성의 일부분으로 훈련
- 매우 고차원의 데이터셋을 다룰 때 유용
  - **랜덤 패치 방식**: 특성과 샘플을 모두 샘플링하는 방식
  - **랜덤 서브스페이스 방식**: 훈련 샘플을 모두 사용하고 특성은 샘플링하는 방식



## 7.4 랜덤 포레스트

- 배깅(또는 페이스팅)을 적용한 결정트리 앙상블

- 전형적으로 `max_samples`를 훈련 세트 크기로 지정

- `RandomForestClassifier`를 사용

  ```bash
  from sklearn.ensemble import RandomForestClassifier
  
  rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
  rnd_clf.fit(X_train, y_train)
  
  y_pred_rf = rnd_clf.predict(X_test)
  ```

  - `DecisionTree Classifier`와 `BaggingClassifier`의 매개변수를 모두 가지고 있음
  - 트리의 노드를 분할할 때 전체 특성 중 최선의 특성을 찾는 대신 무작위로 선택한 특성 후보 중 최적의 특성을 찾는 식으로 무작위성 주입



### 7.4.1 엑스트라 트리

- 트리의 무작위성을 위해 후보 특성을 사용하여 무작위로 분할한 다음 그 중 최상의 분할 선택

- **익스트림 랜덤 트리** 앙상블: 극단적으로 무작위한 트리의 랜덤 포레스트 (엑스트라 트리)

- 편향 up, 분산 down

- 임곗값을 찾을 필요가 없기 때문에 일반적인 랜덤 포레스트보다 엑스트라 트리가 훨씬 빠름

- `ExtraTreesClassifier`를 통해 엑스트라 트리 사용

  

### 7.4.2 특성 중요도

- 랜덤 포레스트: 특성의 상대적 중요도를 측정하기 쉬움
- 사이킷런은 훈련이 끝난 뒤 특성마다 자동으로 이 가중치를 계산하고 중요도의 전체 합이 1이 되도록 결괏값을 정규화
  - `feature_importances_`변수에 가중치 저장
- 랜덤 포레스트는 특성을 선택해야 할 때 어떤 특성이 중요한지 빠르게 확인 가능



## 7.5 부스팅

- **부스팅**: 약한 학습기를 여러 개 연결하여 강한 학습기를 만드는 앙상블 방법
- 앞의 모델을 보완해나가면서 일련의 예측기를 학습
- **에이디부스트**와 **그레디언트 부스팅** 등 존재



### 7.5.1 에이다부스트

- 이전 예측기를 보완하는 새로운 예측기를 만드는 방법: 이전 모델이 과소적합 했던 훈련 샘플의 가중치를 더 높이는 것

  ![image-20210302231708432](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302231708432.png)

- moons데이터셋에 훈련시킨 다섯 개의 연속된 예측기 결정 경계

- 오른쪽 그래프는 학습률을 반으로 낮춘 것만 빼고 똑같은 일련의 예측기를 나타냄

  ![image-20210302231949631](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302231949631.png)

  - 경사하강법과 비슷하게 에이다부스트는 더 좋은 결과를 얻어오기 위해 앙상블에 예측기를 추가

- 모든 예측기가 훈련을 마친 후 배깅이나 페이스팅과 비슷한 방식으로 예측을 만듦

- ![image-20210302232501799](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302232501799.png)

- 위의 에러율이 학습할 때마다 훈련세트에 계산됨

- ![image-20210302232736629](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302232736629.png)

- 예측기가 정확할수록 가중치가 높아짐

- ![image-20210302232817513](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302232817513.png)

- 위 식을 이용하여 샘플의 가중치 업데이트

- 다음 모든 샘플의 가중치 정규화

- 마지막으로 새 예측기가 업데이트된 가중치를 사용하여 훈련되고 전체 과정 반복

- 지정된 예측기 수에 도달하거나 완벽한 예측기가 만들어지면 중지

- 예측을 할 때 에이다부스트는 단순히 모든 예측기의 예측을 계산하고 예측기 가중치를 더해 예측 결과를 만듦

  ![image-20210302233046218](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302233046218.png)

  - 사이킷런은 SAMME라는 에이다부스트의 다중 클래스 버전을 사용

    ```bash
    from sklearn.ensemble import AdaBoostClassifier
    
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    ```



### 7.5.2 그레이디언트 부스팅

- 앙상블에 이전까지의 오차를 보정하도록 예측기를 순차적으로 추가

- 이전 예측기가 만든 잔여오차에 새로운 예측기를 학습

  - 결정트리를 기반예측기로 사용하는 회귀문제(**그레디언트 트리 부스팅** or **그레디언트 부스트디 회귀 트리**)

    ```bash
    from sklearn.tree import DecisionTreeRegressor
    
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)
    ```

    - 잡음이 섞인 2차 곡선 형태의 훈련 세트 훈련

    - 첫번째 예측기에서 생긴 잔여 오차에 두번째 `DecisionTreeRegressor` 훈련

      ```bash
      y2 = y - tree_reg1.predict(X)
      tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
      tree_reg2.fit(X, y2)
      ```

      - 두번째 예측기가 만든 잔여 오차에 세번째 회귀 모델 훈련

        ```bash
        y3 = y2 - tree_reg2.predict(X)
        tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_reg3.fit(X, y3)
        ```

  - 세 개의 트리를 포함하는 앙상블 모델 생성

    ```bash
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    ```

    ![image-20210302234326318](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302234326318.png)

    - 왼쪽 열: 세 트리의 예측 / 오른쪽 열: 앙상블의 예측
    - 첫번째 행은 앙상블에 트리가 하나만 있기 때문에 첫 번째 트리의 예측과 같음
    - 두번째 행에서는 새로운 트리가 첫 번째 트리의 잔여 오차에 대해 학습
    - 세번째 행에서는 또 다른 트리가 두번째 트리의 잔여 오차에 훈련
    - 트리가 앙상블에 추가될수록 앙상블의 예측이 점차 좋아짐

- 사이킷런의 `GradientBoostingRegressor`를 사용하면 GBRT 앙상블을 간단하게 훈련시킬 수 있음

  ```bash
  from sklearn.ensemble import GradientBoostingRegressor
  
  gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
  gbrt.fit(X, y)
  ```

  - `learning_rate` 매개변수가 각 트리의 기여 정도를 조절

  - 낮게 설정하면 많은 트리가 필요하지만 예측의 성능이 좋아짐(축소)

    ![image-20210302235221529](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302235221529.png)

    - 왼쪽은 훈련 세트를 학습하기에 트리가 충분하지 않음
    - 오른쪽은 트리가 너무 많아 훈련 세트에 과대적합

- 최적의 트리 수를 찾기 위해 조기종료 기법 사용 가능 (`staged_predict()` 메서드 사용)

  - 이 메서드는 훈련의 각 단계에서 앙상블에 의해 만들어진 예측기를 순회하는 반복자를 반환

    ```bash
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
    
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)
    
    errors = [mean_squared_error(y_val, y_pred)
              for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors) + 1
    
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)
    ```

    - 위 코드는 120개의 트리로 GPRT 앙상블을 훈련시키고 최적의 트리 수를 찾기 위해 각 훈련 단계에서 검증 오차 측정
    - 마지막에 최적의 트리 수를 사용하여 새로운 GBRT 앙상블 훈련

    ![image-20210302235959250](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210302235959250.png)

  - 실제 훈련을 중지하는 방법으로 조기종료 구현

  - `warm_start=True`로 설정시 사이킷런이 `fit()` 메서드가 호출될 때 기존 트리를 유지하고 훈련을 추가할 수 있도록 해줌

    ```bash
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
    
    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping
    ```

- `GradientBoostingRegressor`는 각 트리가 훈련할 때 사용할 훈련 샘플의 비율을 지정할 수 있는 `subsample`매개변수도 지원 (확률적 그레디언트 부스팅)

- 최적화된 그레디언트 부스팅(XGBoost-익스트림 그레디언트 부스팅)

  - 사이킷런과 아주 비슷
  - 자동 조기 종료와 같은 기능 제공



## 7.6 스태킹

- 예측기를 취합하는 모델을 훈련시키는 방법

  ![image-20210303215537683](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210303215537683.png)

  - 블렌더를 학습시키는 방법은 홀드 아웃 세트를 사용하는 것

    - 훈련 세트를 두 개의 서브셋으로 나눔

    - 첫 번째 서브셋은 첫 번째 레이어의 예측을 훈련시키기 위해 사용됨

      ![image-20210303220020052](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210303220020052.png)

    - 첫 번째 레이어의 예측기를 사용해 두번째 세트에 대한 예측을 만듦

    - 홀드 아웃 세트의 각 샘플에 대해 세 개의 예측값 존재

    - 타깃값은 그대로 쓰고 앞에서 예측한 값을 입력 특성으로 사용하는 훈련 세트를 만들 수 있음

    - 첫 번째 레이어의 예측을 가지고 타깃값을 예측하도록 학습

      ![image-20210303220722988](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210303220722988.png)

    - 블렌더를 여러 개 훈련시키는 것도 가능

    - 블렌더만의 레이어 생성

    - 훈련 세트를 세 개의 서브셋으로 나눔

    - 첫번째 세트는 첫번쨰 레이어를 훈련시키는데 사용, 두번째 세트는 두번째 레이어를 훈련시키기 위한 훈련세트, 세번째 세트는 세번쨰 레이어를 훈련시키기 위한 훈련세트를 만드는 데 사용

      ![image-20210303221235161](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\07_ensemble_learning_and_random_forests\image-20210303221235161.png)

      - 사이킷런에서는 직접 지원 안함





## 7.# 모르는 내용

### 1. 코드



### 2. 내용

### 3. 연구할 부분

- 배깅방식을 이용할 때, 샘플 당 중복 횟수 제한 가능? 제한이 가능하다면 oob평가를 이용하지 않고도 정확도 평가 가능하지않나

