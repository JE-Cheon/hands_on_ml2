# 5장 서포트 백터 머신(SVM)

- **서포트 벡터 머신**: 강력, 선형 or 비선형 분류, 회귀, 이상치 탐색에도 사용할 수 있는 다목적 머신러닝 모델
- 복잡한 분류 문제에 잘 들어맞음



## 5.1 선형 SVM 분류

- **라지 마진 분류**

  ![image-20210310151550413](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310151550413.png)

  - 왼쪽: 점선의 경계가 클래스를 적절하게 분류 못함, 다른 두 경계는 결정 경계가 샘플에 너무 가까워 새로운 샘플에 잘 작동하지 못함
  - 오른쪽: 실선-SVM분류기 결정경계, 클래스를 정확하게 나누고 훈련 샘플로부터 멀리 떨어져있음

- **서포트 벡터**

  ![image-20210310151929654](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310151929654.png)

  - 도로 경계에 위치한 샘플로 결정 경계에 전적으로 영향을 주는 벡터
  - SVM은 특성 스케일에 민감, 사이킷런의 `StandardScaler`를 사용하여 결정경계 조정



### 5.1.1 소프트 마진 분류

- **하드 마진 분류**: 모든 샘플이 도로 바깥쪽에 올바르게 분리되어 있는 경우

  - 단정

    - 데이터가 선형으로 구분될 수 있어야 작동
    - 이상치에 민감

    ![image-20210310154149918](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310154149918.png)

    - 이상치가 생길 경우 하드마진을 찾을 수 없음

  - **소프트 마진 분류**: 도로의 폭을 넓게 유지하는 것과 **마진오류** 사이에 적절한 균형을 잡는 것

- 하이퍼 파라미터 지정에 따른 결정경계의 변화

  - `C`를 낮게 설정할 경우 도로 폭이 넓어지고, 높게 설정할 경우 도로 폭이 좁아짐

    ![image-20210310154855156](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310154855156.png)

    - 왼쪽 모델에 마진오류가 많지만 일반화가 더 잘됨

- 붓꽃 데이터셋을 적재하고 특성 스케일 변경 후 품종을 감지하기 위한 선형 SVM 모델 훈련(힌지 손실 함수를 적용한 `LinearSVC` 클래스 사용)

  ```bash
  import numpy as np
  from sklearn import datasets
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import LinearSVC
  
  iris = datasets.load_iris()
  X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 너비
  y = (iris["target"] == 2).astype(np.float64)  # Iris virginica
  
  svm_clf = Pipeline([
          ("scaler", StandardScaler()),
          ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
      ])
  
  svm_clf.fit(X, y)
  ```

  ```bash
  svm_clf.predict([[5.5, 1.7]])
  ```

  - `LinearSVC`클래스 대신 선형 커널을 사용하는 `SVC`클래스로 대체 가능
  - `SVC(kernel="linear",C=1)`, `SGDClassifier(loss="hinge",alpha=1/(m*C))` 와 같은 모델로 같은 결과 도출 가능
  - `LinearSVC` 만큼 다른 모델이 빠르진 않지만, 데이터셋이 매우 커서 메모리에 적재할 수 없거나, 온라인 학습으로 분류 문제를 다룰 때 유용



## 5.2 비선형 SVM 분류

- 선형적으로 분류할 수 없는 데이터셋의 경우 다항특성과 같은 특성을 추가

- 다항 특성의 추가로 선형적으로 구분되는 데이터셋을 만들 수 있음

  ![image-20210310160607919](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310160607919.png)

  - 왼쪽의 경우 선형적으로 구분할 수 없는 데이터

  - 오른쪽에서 특성을 추가하여 데이터 셋을 선형적으로 구분 가능

  - 사이킷런에서 `PolynomialFeatures`, `StandardScaler`, `LinearSVC`를 연결하여 `Pipeline` 생성

    ```bash
    from sklearn.datasets import make_moons
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    
    polynomial_svm_clf = Pipeline([
            ("poly_features", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
        ])
    
    polynomial_svm_clf.fit(X, y)
    ```

    ![image-20210310162051419](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310162051419.png)



### 5.2.1 다항식 커널

- **커널 트릭**: 실제로 특성을 추가하지 않고, 다항식 특성을 많이 추가한 것과 같은 결과 도출

  - 실제로 어떤 특성도 추가하지 않기 때문에 엄청난 수의 특성 조합이 생기지 않음

    ```bash
    from sklearn.svm import SVC
    
    poly_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
        ])
    poly_kernel_svm_clf.fit(X, y)
    ```

    - 위 코드는 3차 다항식 커널을 사용하여 SVM 분류기 훈련

    ![image-20210310162642825](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310162642825.png)

    - 왼쪽: 3차 다항식 커널
    - 오른쪽: 10차 다항식 커널
    - 매개변수`coef0`모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절

### 5.2.2 유사도 특성

- 유사도 함수: 각 샘플이 특정 **랜드마크**와 얼마나 닮았는지 측정

- 가우시안 **방사 기저 함수**

  ![image-20210310184711641](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310184711641.png)

  - 이 함수는 0(랜드마크에서 많이 떨어진 경우)부터 1(랜드마크와 같은 위치일 경우)까지 변화하며 종모양으로 나타남

    ![image-20210310185036785](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310185036785.png)

    - 방사기저함수를 이용하여 데이터 재배치
    - 재배치 후 선형적 구분 가능
    - 랜드마크는 모든 샘플 위치에 랜드마크 설정하는 것이 가장 간단한 방법
    - 단점: 훈련 세트가 매우 클 경우 동일한 크기의 아주 많은 특성이 만들어짐



### 5.2.3 가우시안 RBF 커널

- 커널 트릭으로 특성 계산 단계를 줄어주는 것이 가능

- 가우시안 RBF 커널을 이용한 SVC 모델

  ```bash
  rbf_kernel_svm_clf = Pipeline([
          ("scaler", StandardScaler()),
          ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
      ])
  rbf_kernel_svm_clf.fit(X, y)
  ```

  ![image-20210310190758418](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310190758418.png)

  - `gamma()`와 `C`를 바꾸어 훈련시킨 모델
  - `gamma`를 증가시키면 종 모양 그래프가 좁아져서 샘플의 영향 범위가 작아짐
  - `gamma`를 감소시키면 결정 경계가 더 부드러워짐

- **문자열 커널**은 텍스트 문서나 DNA 서열을 분류할 때 사용



### 5.2.4 계산복잡도

- `LinearSVC` 파이썬 클래스는 선형 SVM을 위한 최적화된 알고리즘을 구현한 liblinear 라이브러리 기반
- 위 라이브러리는 커널 트릭을 지원하지 않지만 훈련 샘플과 특성수에 거의 선형적으로 늘어남
- 훈련 시간 복잡도 = ***O(m x n)***

- 정밀도를 높이면 알고리즘의 수행시간이 길어짐 => 허용오차 파라미터로 조절
- SVC는 커널 트릭 알고리즘을 구현한 libsvm 라이브러리를 기반으로 함
- 훈련 시간 복잡도는 ***O(m^2 x n)***과 ***O(m^3 x n)*** 사이 => 훈련 샘플 수가 커지면 엄청나게 느려짐
- 작거나 중간 규모의 훈련 세트에 알고리즘이 잘 맞음

![image-20210310192207382](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310192207382.png)



## 5.3 SVM 회귀

- SVM 알고리즘은 회귀에도 사용 가능

- 회귀에 적용하는 방법은 목표를 반대로 하는 것 => 제한된 마진오류 안에서 도로 안에 가능한 많은 샘플이 들어가도록 학습

  ![image-20210310192657650](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310192657650.png)

  - 마진 안에서 훈련 샘플이 추가되어도 모델의 예측에 영향이 없음
  - **오차에 민감하지 않다**

- `LinearSVR`을 사용한 선형 회귀

  ```bash
  from sklearn.svm import SVR
  
  svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
  svm_poly_reg.fit(X, y)
  ```

  ![image-20210310194439497](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310194439497.png)

  - 비선형 회귀 작업을 처리하려면 커널 SVM 모델 사용
  - 왼쪽 그래프는 규제가 거의 없고, 오른쪽 그래프는 규제가 훨씬 많음
  - 계산 복잡도가 훈련세트의 크기에 비례해서 선형적으로 늘어남



## 5.4 SVM 이론



### 5.4.1 결정 함수와 예측

- 선형 SVM 분류기 모델은 단순히 결정함수 ***wtx***+b 를 계산해서 새로운 샘플 **x**의 클래스를 예측

- 결괏값이 0보다 크면 ***y hat***은 양성 클래스, 그렇지 않으면 음성클래스가 됨

  ![image-20210310195729068](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310195729068.png)

  ![image-20210310195752615](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310195752615.png)

  - 결정함수의 형태: 특성이 두 개인 데이터셋이기 때문에 2차원 평면
  - 결정 경계는 결정 함수의 값인 0인 점들로 이루어져있음
  - 점선들은 결정함수 값이 1 또는 -1인 점들을 나타냄, 결정경계와 나란하고 일정한 거리만큼 떨어져서 마진 형성
  - 선형 SVM 분류기를 훈련한다는 것은 마진 오류를 하나도 발생하지 않고나 제한적인 마진 오류를 가지면서 가능한 마진을 크게 하는 가중치와 편향을 찾는 것



### 5.4.2 목적 함수

- 결정 함수의 기울기 = 가중치 백터의 norm

  ![image-20210310201607328](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310201607328.png)

  - 기울기와 마진은 반비례관계

- 마진을 크게 하기 위해 norm w를 최소화 함

  ![image-20210310202014885](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310202014885.png)

- 소프트 마진 분류기의 목적 함수를 구성하기 위해서는 **슬랙 변수** 도입

- 슬랙변수: i번째 샘플이 얼마나 마진을 위반할지를 정함

  - 마진 오류를 최소화하기 위해 가능한 한 슬랙 변수의 값을 작게 만듦

  - 마진을 크게 하기 위해 가중치 norm의 절반 값을 가능한 작게 만듦

    ![image-20210310202926103](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310202926103.png)



### 5.4.3 콰드라틱 프로그래밍

- **콰드라틱 프로그래밍**: 선형적인 제약 조건이 있는 볼록 함수의 이차 최적화 문제

  ![image-20210310203206360](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310203206360.png)

  ![image-20210310203227613](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310203227613.png)

  - 하드 마진 선형 SVM 분류기를 훈련시키는 방법: 준비되어 있는 QP 알고리즘에 관련 파라미터를 전달하는것



### 5.4.4 쌍대 문제

- **원 문제** 라는 제약이 있는 최적화 문제가 주어지면 **쌍대 문제**라고 하는 깊게 관련된 다른 문제로 표현 가능

- SVM 모델로는 원문제와 쌍대문제의 해가 같기 때문에 둘 중 하나를 선택하여 푸는 것 가능

- ![image-20210310211126977](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310211126977.png)

- 이 식을 최소화하는 벡터를 찾으면 원 문제의 식을 최소화하는 가중치와 편향의 예측값을 계산 가능

  ![image-20210310211248929](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310211248929.png)

  - 훈련 샘플 수가 특성 개수보다 작을 때 원 문제보다 쌍대 문제를 푸는 것이 더 빠름
  - 쌍대문제는 원 문제에서 적용이 안되는 커널 트릭을 가능하게 함



### 5.4.5 커널 SVM

- 2차 다항식 매핑 함수

  ![image-20210310212132660](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310212132660.png)

  - 두 개의 2차원 벡터 a와 b에 2차 다항식 매핑 적용 후 변환된 벡터로 점곱 시행

    ![image-20210310212223610](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310212223610.png)

    - 원 백터의 점곱의 제곱과 같은 결과

  - 모든 훈련 샘플에 변환을 적용하면 쌍대문제에 점곱이 포함됨

  - 변환된 벡터의 점곱을 간단하게 바꿀 수 있음 => 훈련 샘플을 변환할 필요가 없음

  - 전체 과정에 필요한 계산량 측면에서 효율적임

- 머신러닝에서 **커널**은 변환을 계산하지 않고 점곱을 계산할 수 있는 함수

  ![image-20210310212614634](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310212614634.png)

  - 머서의 정리
    - 중간 과정을 모르더라도 커널을 계산할 수 있음

- 선형 SVM 분류기일 경우, 쌍대 문제를 풀어 원문제를 해결하는 방법을 알려 줌

- 가중치를 모른 체 예측을 만드는 것이 커널로 가능

  ![image-20210310213408339](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310213408339.png)

  ![image-20210310213439012](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310213439012.png)



### 5.4.6 온라인 SVM

- 온라인 학습: 새로운 샘플이 생겼을 떄 점진적으로 학습하는 것

- 원 문제로부터 유도된 비용함수를 최소화하기 위한 경사하강법을 사용

  ![image-20210310213615670](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\05_support_vector_machine\image-20210310213615670.png)

  - 첫번째 항은 모델이 작은 가중치 벡터를 가지도록 제약을 가해 마진을 크게 만듦
  - 두번째 항은 모든 마진 오류를 계산
  - 어떤 샘플이 도로에서 올바른 방향으로 벗어나있다면 마진오류는 0



- 힌지손실

  - 서브 그레디언트를 사용하여 경사 하강법 사용 가능

    



## 5.# 모르는 부분

### 1. 내용

- SVM에서 결정경계를 정하는 방법
- 서포트벡터는 왜 결정 경계에 직접적인 영향을 주는가?
- 비선형 SVM 분류에서, 다항 특성을 추가하는 건 경계가 아니라 데이터 자체?
- 유사도 특성에서 말하는 랜드마크는 어떤 것?
- SVM회귀모델은 왜 마진에 민감하지 않은가?
- SVM회귀모델로 하려는 것이 뭔지 잘 이해 안됨
- 쌍대문제가 의미하는 것
- 원문제와 쌍대문제에서 왜 쌍대문제가 더 빨리 풀 수 있는가?
- 

### 2. 코드

### 3. 연구할 부분



