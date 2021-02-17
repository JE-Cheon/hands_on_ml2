# 4장 모델 훈련

- 어떻게 작동하는지 이해하면 적절한 모델, 올바른 훈련 알고리즘, 작업에 맞는 좋은 하이퍼 파라미터를 빠르게 찾는 것 가능
- 신경망 이해 및 구축, 훈련에 필수



## 4.1 선형 회귀

- 입력 특성의 가중치 합과 편향(절편)이라는 상수를 더해서 예측을 만듦

  ![image-20210211183943784](C:\Users\sma05\AppData\Roaming\Typora\typora-user-images\image-20210211183943784.png)
  - *y hat*: 예측값

  - *n*: 특성의 수

  - *xi*: i번째 특성값

  - *theta j*: j번째 모델 파라미터(편향+가중치)

  - 벡터형태 표현

    ![image-20210211184241286](C:\Users\sma05\AppData\Roaming\Typora\typora-user-images\image-20210211184241286.png)

    - **theta**: 편향과 가중치를 담은 모델의 파라미터 벡터
    - **x**: 샘플의 특성 벡터, *x0*는 항상 1
    - **theta*x**: 백터 **theta**와 **x**의 점곱 (식 4-1 형태와 같음)
    - *h theta*: 모델 파라미터 **theta**를 사용한 가설 함수

- 모델 훈련: 모델이 훈련 세트에 가장 잘 맞도록 모델 파라미터를 설정하는 것

  - 모델이 데이터에 얼마나 잘 들어맞는지 측정

  - RMSE(평균 제곱근 오차)를 최소화하는 **theta**를 설정

  - MSE를 최소화 하는 것이 위와 같은 결과를 냄

    ![image-20210212040133088](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210212040133088.png)

### 4.1.1 정규방정식

- 정규방정식: 비용함수를 최고화하는 **theta**값을 찾기 위한 해석적인 방법

  ![image-20210212040337249](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210212040337249.png)

  - **theta hat**은 비용함수를 최소화하는 **theta**값

  - **y**는 *y(1)*부터 *y(m)*까지 포함하는 타깃 벡터

  - 선형데이터생성

    ```bash
    import numpy as np
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    ```

    ![image-20210212040831896](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210212040831896.png)

  - 정규 방정식을 이용한 **theta hat** 계산

    ```bash
    X_b = np.c_[np.ones((100, 1)), X]  # 모든 샘플에 x0 = 1을 추가합니다.
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    ```

  - 확인

    ```bash
    theta_best
    ```

    - 비슷한 파라미터를 만들었지만 잡음때문에 정확한 재현 불가능

  - **theta hat** 을 사용한 예측(p.163)

    ```bash
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # 모든 샘플에 x0 = 1을 추가합니다.
    y_predict = X_new_b.dot(theta_best)
    y_predict
    ```

  - 그래프 표현

    ```bash
    plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.show()
    ```

  - 사이킷런을 통한 선형 회귀
  
    ```bash
    from sklearn.linear_model import LinearRegression
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_reg.intercept_, lin_reg.coef_
    ```
  
    ```bash
    lin_reg.predict(X_new)
    ```
  
    - `LinearRegression`클래스는 `scipy.linalg.lstsq()`함수 호출 가능
  
      ```bash
      # 싸이파이 lstsq() 함수를 사용하려면 scipy.linalg.lstsq(X_b, y)와 같이 씁니다.
      theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
      theta_best_svd
      ```
  
      ![image-20210214004847951](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214004847951.png)
  
      - **X+**는 **X**의 유사역행렬
  
      - `np.linalg.pinv()` 함수로 유사역행렬 구하는 것 가능
  
        ```bash
        np.linalg.pinv(X_b).dot(y)
        ```

### 4.1.2 계산 복잡도

- *(n+1)x(n+1)* 크기가 되는 **XtX**의 역행렬 계산
- 특성 수가 늘어날 수록 계산 시간이 제곱으로 늘어남

## 4.2 경사 하강법

- 여러 종류의 문제에서 최적의 해법을 찾을 수 있는 일반적인 최적화 알고리즘

- 비용함수를 최소화하기 위해 반복해서 파라미터를 조정하는 것

- 그레디언트가 0이 되도록 하는 것이 비용함수를 최소화 하는 것

- **theta**를 임의의 값으로 시작해서 비용함수가 감소되는 방향으로 진행

  ![image-20210214012023877](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214012023877.png)

  - 스텝의 크기가 중요. 학습률 하이퍼파라미터로 결정됨

  - 학습률이 너무 작을 경우 반복을 많이 진행해야 하기 때문에 시간이 오래 걸림

    ![image-20210214012124527](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214012124527.png)

  - 학습률이 너무 크면 골짜기를 가로질러 반대편으로 건너뛰어 이전보다 더 높은 곳으로 올라갈 가능성이 있음

    ![image-20210214012201853](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214012201853.png)

  - 모든 비용함수가 매끄럽진 않음

  - 왼쪽에서 시작하면 **전역 최솟값**보다 **지역 최솟값**에 수렴

  - 오른쪽에서 시작하면 평탄한 지역을 지나기 위해 시간이 오래 걸리고 평지에서 멈춰 최솟값에 도달하지 못함

    ![image-20210214012355424](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214012355424.png)

  - 선형회귀를 위한 MSE 비용함수는 **볼록함수**

  - 지역 최솟값이 없고 하나의 전역 최솟값 존재, 연속함수, 기울기가 갑자기 변하지 않음 -> 경사하강법을 통해 전역 최솟값에 가깝게 접근할 수 있음

  - 특성들의 스케일에 따라 다른 비용함수 형태를 가짐

    ![image-20210214012624885](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214012624885.png)

    - 왼쪽의 경사하강법 알고리즘이 최솟값으로 곧장 진행하여 빠르게 도달
    - 오른쪽 그래프는 전역 최솟값에 빠르게 도달하다가 평평해지며 오랜 시간이 걸림
    - 경사 하강법을 사용할 때 모든 특성의 스케일을 같게 만들어주기 위해 `StandardScaler` 사용

  - 최적의 파라미터 조합을 찾는 것 -> **파라미터 공간**에서 찾음



### 4.2.1 배치 경사 하강법

- 경사 하강법을 구현하기 위해서는 각 모델 파라미터에 대해 비용함수의 그레디언트를 계산 -> 파라미터의 변화에 따른 비용함수의 변화 계산(**편도함수**)

  ![image-20210214013114590](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214013114590.png)

  - 비용함수의 그레디언트 벡터(매 경사 하강법 스텝에서 전체 훈련 세트 **X**에 대해 계산)

    ![image-20210214013155809](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214013155809.png)

  - 위로 향하는 그레디언트 벡터가 구해지면 반대방향으로 내려가야 함

  - **theta**에서 MSE의 편도함수를 때야 함

  - 스텝의 크기를 정하기 위해 그레디언트 벡터에 학습률을 곱함

    ![image-20210214015324434](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214015324434.png)

    ```bash
    eta = 0.1  # 학습률
    n_iterations = 1000
    m = 100
    
    theta = np.random.randn(2,1)  # 랜덤 초기화
    
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    ```

    - 학습률 변화에 따른 경사하강법

      ![image-20210214015627971](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214015627971.png)
      - 왼쪽은 학습률이 너무 낮음
      - 가운데가 적당한 학습률
      - 오른쪽은 학습률이 높음

    - 적절한 학습률 탐색을 위해 `GridSearch` 사용

      - 너무 오래 걸리지 않게 하기 위해 반복 횟수를 제한

    - 반복 횟수 지정 방법

      - 반복 횟수를 크게 지정하고 그레디언트벡터가 작아지는 경우
      - 벡터의 노름이 어떤 값(허용오차)보다 작아지는 경우



### 4.2.2 확률적 경사 하강법

- 배치 경사 하강법은 매 스텝에 전체 훈련 세트를 이용하여 그레디언트를 계산

- **확률적 경사 하강법**: 매 스텝에서 한 개의 샘플을 무작위로 선택하고 그 하나의 샘플에 대한 그레디언트를 계산

  ![image-20210214022906370](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214022906370.png)

- 장점

  - 하나의 샘플만 처리하면 되기 때문에 알고리즘이 빠름
  - 매 반복에서 하나의 샘플만 메모리에 있으면 되기 때문에 매우 큰 훈련 세트도 훈련 가능
  - 비용함수가 매우 불규칙할 때 지역최솟값을 건너뛰도록 도와줌
  - 무작위성으로 지역 최솟값에서 탈출 가능

- 단점

  - 확률적이기 때문에 배치 경사 하강법보다 훨씬 불안정

  - 부드럽게 감소하지 않고 요동치며 평균적으로 감소

  - 최적의 알고리즘을 찾는 데에 어려움이 있음

  - 무작위성은 지역 최솟값에서 탈출시켜주지만 알고리즘을 전역 최솟값에 다다르지 못한다는 단점이 있음

  - 극복방법

    - 학습률을 점진적으로 감소시킴(**담금질 기법 알고리즘**)
    - 학습 스케줄: 반복에서 학습룰을 결정하는 함수
    - 학습률이 너무 빠르게 줄어들면 지역 최솟값에 갇히거나 최솟값까지 가는 중간에 멈춰버림
    - 학습률이 너무 천천히 줄어들면 오랫동안 최솟값 주변을 맴돌거나 훈련을 일찍 중지해서 지역 최솟값에 머무를 수 있음

  - 코드 구현

    ```bash
    n_epochs = 50
    t0, t1 = 5, 50  # 학습 스케줄 하이퍼파라미터
    
    def learning_schedule(t):
        return t0 / (t + t1)
    
    theta = np.random.randn(2,1)  # 랜덤 초기화
    
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradient
    ```

    - 한 반복에서 m번 되풀이(**에포크**)

      ![image-20210214023423195](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214023423195.png)

  - 샘플을 무작위로 선택하기 때문에 어떤 샘플이 한 에포크에서 여러 번 선택될 수 있고, 선택되지 못할 수 있음

  - 훈련 세트를 섞은 후 차례대로 하나씩 선택하고 다음 에포크에서 다시 섞는 식의 방법 사용

  - 사이킷런에서 SGD 방식으로 선형회귀를 사용하기 위해서는` SGDRegressor` 클래스 사용

  - 코드 구현

    ```bash
    from sklearn.linear_model import SGDRegressor
    
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())
    ```

    ```bash
    sgd_reg.intercept_, sgd_reg.coef_
    ```



### 4.2.3 미니배치 경사 하강법

- **미니배치**라 부르는 임의의 작은 샘플 세트에 대해 그레디언트를 계산하는 방법

- GPU를 사용하기 때문에 성능적으로 향상

- 미니배치를 어느정도 크게 할 경우 SGD보다 덜 불규칙하게 움직임

- SGD보다 최솟값에 더 가까이 도달하지만 지역 최솟값에서 빠져나오기 힘들수도 있음

- 세가지 방법으로 파라미터 공간에서 움직인 경로

  ![image-20210214024548276](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214024548276.png)

  - 배치 경사 하강법은 실제 최솟값에서 멈춤, 다른 방법은 주변을 맴돈다
  - 배치 경사 하강법은 매 스텝에서 많은 시간 소요
  - 확률적 경사 하강법과 미니배치 경사 하강법도 적절한 학습 스케줄을 사용하면 최솟값에 도달

- 알고리즘 선형 회귀 비교

  ![image-20210214025042032](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214025042032.png)



## 4.3 다항 회귀

- 비선형 데이터를 학습하는데에 선형모델 사용 가능

- 다항회귀: 각 특성의 거듭제곱을 새로운 특성으로 추가하고, 이 확장된 특성을 포함한 데이터셋에 선형 모델을 훈련시키는 것

- 2차방정식 비선형 데이터 생성

  ```bash
  m = 100
  X = 6 * np.random.rand(m, 1) - 3
  y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
  ```

  ![image-20210214025426582](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210214025426582.png)

  - 직선으로는 맞지 않는 데이터

  - 사이킷런의 `PolynomialFeatures` 클래스를 사용하여 훈련 데이터 변환

  - 훈련 세트에 있는 각 특성을 제곱하여 새로운 특성으로 추가

    ```bash
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    X[0]
    ```

    ```bash
    X_poly[0]
    ```

    - 확장된 훈련 데이터에 `LinearRegression` 적용

      ```bash
      lin_reg = LinearRegression()
      lin_reg.fit(X_poly, y)
      lin_reg.intercept_, lin_reg.coef_
      ```

  - 특성이 여러 개일 때 이 특성 사이의 관계를 찾을 수 있음
  - `PolynomialFeatures` 가 주어진 차수까지 특성 간의 모든 교차항을 추가하기 때문



## 4.4 학습 곡선

- 고차 다항회귀를 적용하면 보통의 선형회귀에서보다 훨씬 더 훈련 데이터에 잘 맞추려고 할 것

  ![image-20210215165402773](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210215165402773.png)

  - 고차 다항 회귀 모델은 심각하게 훈련데이터에 과대적합
  - 선형모델은 과소적합
  - 2차 다항회귀가 가장 일반화가 잘 되어있음

- 모델의 일반화 성능을 추정하기 위해 교차검증 사용

- 성능이 좋아도 교차검증점수가 나쁘면 모델이 과대적합된 것

- 양 쪽 다 좋지 않으면 과소적합

- 학습곡선 분석을 통해 훈련 성능 측정

  ```bash
  from sklearn.metrics import mean_squared_error
  from sklearn.model_selection import train_test_split
  
  def plot_learning_curves(model, X, y):
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
      train_errors, val_errors = [], []
      for m in range(1, len(X_train)):
          model.fit(X_train[:m], y_train[:m])
          y_train_predict = model.predict(X_train[:m])
          y_val_predict = model.predict(X_val)
          train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
          val_errors.append(mean_squared_error(y_val, y_val_predict))
  
      plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
      plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
  ```

  ```bash
  lin_reg = LinearRegression()
  plot_learning_curves(lin_reg, X, y)
  ```

  ![image-20210216003819021](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216003819021.png)

  - 훈련 데이터의 성능
    - 그래프가 0에서 시작하기 때문에 훈련 세트에 하나 혹은 두 개의 샘플이 있을 땐 모델이 완벽하게 작동
    - 훈련 세트에 샘플이 추가됨에 따라 잡음도 있고 비선형이기 때문에 모델이 훈련 데이터를 완벽히 학습하는게 불가능
    - 평평해질때까지 오차가 상승
    - 이 때부터는 훈련세트에 샘플이 추가돼도 오차 변동 없음
  - 검증데이터의 모델 성능
    - 모델이 적은 수의 훈련 샘플로 훈련될 때 일반화될 수 없어서 검증 오차가 초기에 큼
    - 모델에 훈련 샘플이 추가됨에 따라 학습이 되고 검증오차가 천천히 감소
    - 선형 회귀 직선은 데이터를 잘 모델링 할 수 없어 오차의 감소가 완만해져 훈련세트의 그래프와 가까워짐
  - 이 학습 곡선이 과소적합 모델의 전형적인 모습

- 같은 데이터의 10차 다항 회귀 모델 학습 곡선

  ```bash
  from sklearn.pipeline import Pipeline
  
  polynomial_regression = Pipeline([
          ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
          ("lin_reg", LinearRegression()),
      ])
  
  plot_learning_curves(polynomial_regression, X, y)
  ```

  ![image-20210216004459873](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216004459873.png)

  - 훈련 데이터의 오차가 선형 회귀 모델보다 훨씬 낮음
  - 두 곡선 사이에 공간이 있음: 훈련 데이터에서의 모델 성능이 검증 데이터에서보다 훨씬 나음(과대적합). 더 큰 훈련세트를 사용하면 두 곡선이 점점 가까워짐
  - 편향/분산 트레이드오프
    - 모델의 일반화 오차는 세 가지 다른 종류의 오차의 합으로 표현 가능
    - 편향: 잘못된 가정
    - 분산: 훈련 데이터에 있는 작은 변동에 모델이 과도하게 민감하기 때문에 나타남
    - 줄일 수 없는 오차: 데이터 자체에 있는 잡음
    - 모델의 복잡도가 커지면 통상적으로 분산이 늘어나고 편향이 줄어듦, 모델의 복잡도가 줄어들면 편향이 커지고 분산이 작아짐(트레이드오프)

## 4.5 규제가 있는 선형 모델

- 과대적합을 감소시키는 좋은 방법은 모델을 규제하는 것
- 다항회귀 모델의 규제는 다항식의 차수를 감소시키는 것



### 4.5.1 릿지 회귀

- 규제가 추가된 선형 회귀 버전

- 규제항이 비용함수에 추가

- 학습 알고리즘을 데이터에 맞추고, 모델의 가중치가 가능한 한 작게 유지되도록 노력

- 훈련하는 동안에만 비용함수에 추가됨

- 훈련이 끝나면 모델의 성능을 규제가 없는 성능 지표로 평가

- 하이터파라미터 *alpha*는 모델을 얼마나 많이 규제할지 조절

- *alpha*=0 : 선형회귀와 같음

- *alpha*가 아주 크면 모든 가중치가 0에 가까워지고 데이터의 평균을 지나는 수평선이 됨

  ![image-20210216011450190](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216011450190.png)

  - 편향 *theta0*은 규제 x

- 릿지모델 예시

  ![image-20210216012218252](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216012218252.png)

  - 평범한 릿지 모델(왼쪽)
  - `PolynomialFeatures(degree=10)`을 사용하여 데이터 확장 후 `StandardScaler` 사용하여 스케일 조정 후 릿지 모델 적용(오른쪽)
  - *alpha*를 증가시킬수록 직선에 가까워짐(분산이 줄고 편향이 커짐)

- 선형회귀와 마찬가지로 릿지 회귀를 계산하기 위해 정규방정식이나 경사하강법 사용 가능

- 릿지회귀의 정규방정식

  ![image-20210216013136095](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216013136095.png)

- 사이킷런에서 정규방정식을 사용한 릿지 회귀 적용 예시

  ```bash
  from sklearn.linear_model import Ridge
  ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
  ridge_reg.fit(X, y)
  ridge_reg.predict([[1.5]])
  ```

  - 행렬 분해 사용

- 확률적 경사하강법 사용

  ```bash
  sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
  sgd_reg.fit(X, y.ravel())
  sgd_reg.predict([[1.5]])
  ```

  - `penalty` 매개변수가 사용할 규제 지정
  - `"l2"` : 릿지회귀와 같음



### 4.5.2 라쏘 회귀

- 선형 회귀의 또 다른 규제된 버전

- 비용 함수에 규제항을 더하지만 norm의 제곱을 2로 나는 것 대신 가중치벡터의 norm을 사용

  ![image-20210216014319778](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216014319778.png)

  ![image-20210216014407025](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216014407025.png)

  - 덜 중요한 특성의 가중치 제거가 목표
  - 오른쪽 그래프에서 점선이 3차방정식처럼 보임 -> 차수가 높은 다항 특성의 가중치가 0이 됨
  - 자동으로 특성 선택을 하고 **최소 모델**을 만듦

- 릿지와의 차이점

  - 파라미터가 전역 최적점에 가까워질수록 그레디언트가 작아짐
  - 경사하강법이 자동으로 느려짐
  - *alpha*를 증가시킬수록 최적의 파라미터가 원점에 가까워짐

- 라쏘의 비용함수는 파라미터가 0일 때 미분 불가능

- 서브그레디언트 백터로 경사하강법 적용하는데에 문제 해결

  ![image-20210216020757741](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216020757741.png)

  ```bash
  from sklearn.linear_model import Lasso
  lasso_reg = Lasso(alpha=0.1)
  lasso_reg.fit(X, y)
  lasso_reg.predict([[1.5]])
  ```



### 4.5.3 엘라스틱넷

- 릿지 회귀와 라쏘 회귀를 절충한 모델

- 규제항: 릿지와 회귀의 규제항을 더함

- 혼합정도는 *r*로 조절

  ![image-20210216021242291](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216021242291.png)

- 선형회귀, 릿지, 라쏘 용도

  - 어느정도의 규제는 필요하기 때문에 선형회귀는 피함
  - 릿지가 기본, 쓰이는 특성이 몇 개 뿐이라고 의심되면 라쏘나 엘라스틱넷
  - 특성 수가 훈련 샘플 수보다 많거나 특성 몇 개가 강하게 연관되어 있을 때 엘라스틱넷 선호

  ```bash
  from sklearn.linear_model import ElasticNet
  elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
  elastic_net.fit(X, y)
  elastic_net.predict([[1.5]])
  ```

- 규제 선택법
  
  - 데이터 분포도의 정규화, 스케일링, 인코딩에 중점

### 4.5.4 조기 종료

- 검증 에러가 최솟값에 도달하면 훈련을 중지시키는 방법

- 에포크가 진행됨에 따라 알고리즘이 점차 학습되어 훈련세트에 대한 예측 에러와 점증 세트에 대한 예측 에러가 줄어듦

- 감소하던 검증 에러가 멈췄다가 다시 상승 -> 모델이 훈련 데이터에 과대적합하기 시작

- 검증에러가 최소에 도달하는 즉시 훈련을 멈추는 방법

  ```bash
  from copy import deepcopy		# 책이랑 import가 다른 이유
  
  poly_scaler = Pipeline([
          ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
          ("std_scaler", StandardScaler())
      ])
  
  X_train_poly_scaled = poly_scaler.fit_transform(X_train)
  X_val_poly_scaled = poly_scaler.transform(X_val)
  
  sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                         penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)
  
  minimum_val_error = float("inf")
  best_epoch = None
  best_model = None
  for epoch in range(1000):
      sgd_reg.fit(X_train_poly_scaled, y_train)  # 중지된 곳에서 다시 시작합니다
      y_val_predict = sgd_reg.predict(X_val_poly_scaled)
      val_error = mean_squared_error(y_val, y_val_predict)
      if val_error < minimum_val_error:
          minimum_val_error = val_error
          best_epoch = epoch
          best_model = deepcopy(sgd_reg)
  ```



## 4.6 로지스틱 회귀

- 회귀 알고리즘을 분류에서도 사용 가능
- 샘플이 특정 클래스에 속할 확률을 추정하는 데 사용됨
- 추정 확률이 50%가 넘으면 그 샘플이 해당 클래스에 속한다고 예측(이진분류기 사용)



### 4.6.1 확률 추정

- 입력 특성의 가중치 합 계산

- 바로 결과 출력 x, 특성의 가중치 합 계산

- 결괏값의 로지스틱 출력

  ![image-20210216023201125](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216023201125.png)

  - 로지스틱은 0과 1 사이의 값을 출력하는 시그모이드 함수

    ![image-20210216023242213](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216023242213.png)

    ![image-20210216023255513](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216023255513.png)

  - 로지스틱 회귀 모델이 샘플 **x**가 양성 클래스에 속할 확률을 추정하면 y의 예측값을 쉽게 구함

    ![image-20210216023452989](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216023452989.png)



### 4.6.2 훈련과 비용 함수

- 양성 심플에 대해서는 높은 확률을 추정, 음성 샘플에 대해서는 낮은 확률 추정하는 모델의 파라미터 벡터를 찾는 것이 목표

  ![image-20210216023804441](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216023804441.png)

  - *t*가 0에 가까워지면 *-log(t)*가 매우 커져 타당함
  - *t*가 1에 가까우면 *-log(t)*는 0에 가까워짐

- 전체 훈련 세트에 대한 비용함수는 모든 훈련 샘플의 비용을 평균한 것(로그손실)

  ![image-20210216024755883](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216024755883.png)

  - 최솟값을 계산하는 알려진 해 존재x

  - 볼록함수이기 때문에 경사하강법과 같은 방법으로 최솟값 찾는 것 가능

  - 비용함수의 편미분값

    ![image-20210216024943601](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216024943601.png)

    - 각 샘플에 대해 예측 오차를 계산하고 *j*번째 특성값을 곱해서 모든 훈련 샘플에 대해 평균을 냄
    - 모든 편도함수를 포함한 그레디언트 백터를 만들면 배치 경사 하강법 알고리즘 사용 가능



### 4.6.3 결정 경계

- 붓꽃데이터셋 이용

  ```bash
  from sklearn import datasets
  iris = datasets.load_iris()
  list(iris.keys())
  ```

  ```bash
  X = iris["data"][:, 3:]  # 꽃잎 너비
  y = (iris["target"] == 2).astype(np.int)  # Iris virginica이면 1 아니면 0
  ```

- 로지스틱 회귀 모델 훈련

  ```ba
  from sklearn.linear_model import LogisticRegression
  log_reg = LogisticRegression(solver="lbfgs", random_state=42)
  log_reg.fit(X, y)
  ```

- 꽃잎의 너비가 0~3cm 인 꽃에 대해 모델의 추정 확률 계산

  ```bash
  X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
  y_proba = log_reg.predict_proba(X_new)
  
  plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
  plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
  ```

  ![image-20210216025810625](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216025810625.png)

  - Iris-Verginica(삼각형)의 꽃잎의 너비는 1.4~2.5cm에 분포
  - 다른 붓꽃은 0.1~1.8cm에 분포
  - 양쪽의 확률이 같아지는 1.6cm 근방에서 결정 경계가 만들어짐

- 꽃잎 너비, 꽃잎 길이에 대한 관계

  ![image-20210216030220565](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216030220565.png)

  - 훈련이 끝나면 로지스틱 회귀 분류기가 이 특성을 기반으로 새로운 꽃이 Iris-Verginica인지 확률 추정
  - 점선은 모델이 50% 확률을 추정하는 지점
  - 선을 기준으로 붓꽃으로 판단

- 로지스틱 회귀도 패널티를 통한 규제 가능



### 4.6.4 소프트맥스 회귀

- 로지스틱 회귀는 여러 개의 이진분류기를 훈련시켜 연결하지 않고 직접 다중 클래스 지원 일반화하는 것

- 샘플 **x**가 주어지면 소프트맥스 회귀 모델이 각 클래스 *k*에 대한 점수 *s(**x**)*계산, 그 점수에 소프트맥스 함구를 적용하여 각 클래스의 확률 추정

  ![image-20210216030840520](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216030840520.png)

  - 각 클래스는 자신만의 파라미터 백터 **theta**존재
  - 이 벡터들이 파라미터 행렬의 행으로 저장

- 샘플에 대해 각 클래스 점수가 계산되면 소프트맥스 함수를 통과시켜 클래스에 속할 확률 추정 가능

- 각 점수에 지수함수를 적용한 후 정규화(로그-오즈)

  ![image-20210216031105791](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216031105791.png)

  - **K**는 클래스 수
  - **s(x)**는 샘플에 대한 각 클래스의 점수를 담은 백터
  - **sigma(s(x))**는 샘플에 대한 각 클래스의 점수가 주어졌을 떄 이 샘플이 클래스에 속할 추정 확률

- 추정확률이 가장 높은 클래스를 선택

  ![image-20210216031308633](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216031308633.png)

  - `argmax`연산은 함수를 최대화하는 변수의 값 반환

- 모델이 타깃 클래스에 대해서 높은 확률을 추정하도록 만드는 것이 목적

- **크로스 엔트로피** 비용 함수를 최소화 하는 것이 목적 달성에 기여

  ![image-20210216031515186](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216031515186.png)

  - 크로스 엔트로피는 추정된 클래스의 확률이 타깃 클래스에 얼마나 잘 맞는지 측정하는 용도로 사용

  - 그레디언트 벡터

    ![image-20210216031714736](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\image-20210216031714736.png)

- 각 클래스에 대한 그레디언트 벡터 계산 가능, 비용함수를 최소화하기 위한 파라미터 행렬을 찾기 위해 경사 하강법 알고리즘 사용 가능

  ```bash
  X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 너비
  y = iris["target"]
  
  softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
  softmax_reg.fit(X, y)
  ```

  - `LogisticRegression`은 클래스가 둘 이상일 때 일대다 전략 사용

  - `multi_class` 매개변수를 `"multinomial"`로 바꾸면 소프트맥스 회귀 사용 가능

  - `solver` 매개변수에 `"lbfgs"`로 지정(소프트맥스회귀)

  - 예측값 확인

    ```bash
    softmax_reg.predict([[5, 2]])
    softmax_reg.predict_proba([[5, 2]])
    ```

    



## 4.# 모르는 부분

### 1. 코드

- `np.array` 부분 이해 안됨

  ```bash
  X_new = np.array([[0], [2]])
  X_new_b = np.c_[np.ones((2, 1)), X_new]  # 모든 샘플에 x0 = 1을 추가합니다.
  y_predict = X_new_b.dot(theta_best)
  y_predict
  ```

  - 0과 2를 가진 열백터 생성

- `lin_reg.intercept_`: y절편 (편향)

- `lin_reg.coef_`: 계수 (특성)


### 2. 내용

- 정규방정식의 원리와 각 변수가 뜻하는 부분
  
  - 미분 또는 편미분을 통해 수학적으로 모수를 구하는 공식
  
- 가우시안 잡음
  - 신호의 불규칙한 요동으로 발생
  - 정규분포를 갖는 백색 잡음
  
- MSE 비용함수의 형태가 볼록함수인 이유

  - 편향과 가중치로 인해 정리한 형태가 2차함수 형태

    ![캡처](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\04_training_linear_model\캡처.PNG)

- 회귀를 통해 나오는 array값

  - 구하고자 하는 하이퍼파라미터

### 3. 연구할 부분

- 학습률, 학습 스케줄 만드는 방식