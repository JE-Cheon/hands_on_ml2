# 3장 분류

## 3.1 MNIST

- MNIST 데이터셋: 고등학생과 미국 인구조사국 직원들이 손으로 쓴 70000개의 작은 숫자이미지

- 머신러닝 분야의 'Hello World'

- MNIST 내려받는 코드

  ```python
  from sklearn.datasets import fetch_openml
  mnist = fetch_openml('mnist_784', version=1)
  mnist.keys()
  ```

  - 구조

    - 데이터셋을 설명하는 DCSCR 키
    - 샘플이 하나의 행, 특성이 하나의 열로 구성된 배열을 가진 data 키
    - 레이블 배열을 담은 target 키

  - 배열

    ```python
    X, y = mnist["data"], mnist["target"]
    X.shape
    ```

    ```python
    y.shape
    ```

  - 이미지 확인(`imshow()` 함수 사용)

    ```python
    %matplotlib inline
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    some_digit = X[0]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    plt.axis("off")
    
    save_fig("some_digit_plot")
    plt.show()
    ```

    ```python
    y[0]	# 문자열 레이블을 가짐
    ```

    ```python
    y = y.astype(np.uint8)
    ```

  - 데이터 이미지 샘플

    ![image-20210206014226697](C:\Users\sma05\AppData\Roaming\Typora\typora-user-images\image-20210206014226697.png)

  - 데이터 전처리: 테스트 세트 따로 분리, MNIST 데이터셋은 이미 훈련세트와 테스트세트로 나누어짐

    ```python
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    ```

    - 훈련 세트가 이미 섞여 있기 때문에 모든 교차 검증 폴드를 비슷하게 만듦
    - 데이터셋을 섞을 경우 비슷한 데이터가 모여있는 것을 방지



## 3.2 이진 분류기 훈련

- 이진 분류기: MNIST 데이터셋에서 숫자가 맞는지 아닌지 식별하는 것

  - 분류작업을 위한 타깃 벡터

    ```python
    y_train_5 = (y_train == 5)	# 5는 True고, 다른 숫자는 모두 False
    y_test_5 = (y_test == 5)
    ```

  - 확률적 경사 하강법(SGD): 사이킷런의 SGDClassifier 클래스 사용, 매우 큰 데이터셋을 효율적으로 처리하는 장점

    ```python
    from sklearn.linear_model import SGDClassifier
    
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    ```

    - 무작위성을 이용하기 떄문에 `random_state` 매개변수 지정
    - `max_iter`와 `tol` 같은 일부 매개변수는 사이킷런 다음 버전에서 기본값이 바뀜

  - 이 모델을 통한 숫자 5의 이미지 감지

    ```python
    sgd_clf.predict([some_digit])
    ```

## 3.3 성능 측정

### 3.3.1 교차 검증을 사용한 정확도 측정

- 교차 검증

  - 사이킷런의 `cross_val_score()` 이용

  - 조금 더 제어가 필요하면 직접 구현 필요

    ```python
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    
    # shuffle=False가 기본값이기 때문에 random_state를 삭제하던지 shuffle=True로 지정하라는 경고가 발생합니다.
    # 0.24버전부터는 에러가 발생할 예정이므로 향후 버전을 위해 shuffle=True을 지정합니다.
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]
    
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    ```

    - `StratifiedKFold`: 클래스별 비율이 유지되도록 폴드는 만들기 위해 계층적 샘플링 수행

  - `cross_val_score()` 함수로 폴드가 3개인 k-겹 교차 검증 사용 SGDClassifier 모델 평가

    ```python
    from sklearn.model_selection import cross_val_score
    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    ```

    - 모든 교차 검증 폴드에 대해 정확도가 95% 이상

  - 모든 이미지가 '5 아님' 클래스로 분류하는 더미분류기 생성

    ```python
    from sklearn.base import BaseEstimator
    class Never5Classifier(BaseEstimator):
        def fit(self, X, y=None):
            pass
        def predict(self, X):
            return np.zeros((len(X), 1), dtype=bool)
    ```

    ```python
    never_5_clf = Never5Classifier()
    cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    ```

  - 정확도를 분류기의 성능 측정 지표로 선호하지 않음(불균형 데이터셋을 다룰 때)

### 3.3.2 오차 행렬

- 분류기의 성능을 평가하는 좋은 방법

- 클래스 A의 샘플이 클래스 B로 분류된 횟수를 세는 것

- 먼저 예측값을 만들어야 함

- `cross_val_predict()` 함수 사용

  ```python
  from sklearn.model_selection import cross_val_predict
  
  y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
  ```

  - 평가점수를 반환하지 않고 테스트 폴드에서 얻은 예측을 반환(훈련 세트의 모든 샘플에 대해 깨끗한 예측을 얻게 됨)

- `confusion_matrix()` 함수를 사용하여 오차행렬 생성

  ```python
  from sklearn.metrics import confusion_matrix
  
  confusion_matrix(y_train_5, y_train_pred)
  ```

  - 행: 실제 클래스/ 열: 예측한 클래스
  - 완벽할 경우 대각행렬에만 숫자 존재

- 양성 예측의 정확도를 통해 정보 확인 가능(정밀도)

  ![image-20210207230815574](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\image-20210207230815574.png)

  - TP: 진짜 양성의 수 / FP: 거짓 양성의 수

  - 재현율과 같은 지표까지 함께 사용

  - 재현율: 분류기가 정확하게 감지한 양성샘플의 비율(민감도, 진짜 양성 비율)

    ![image-20210207231026736](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\image-20210207231026736.png)

    - FN: 거짓 음성의 수

  - 오차 행렬 정리

  ![image-20210207231107647](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\image-20210207231107647.png)



### 3.3.3 정밀도와 재현율

- 정밀도와 재현율을 포함하여 분류기의 지표를 계산하는 여러 함수 제공

  ```python
  from sklearn.metrics import precision_score, recall_score
  
  precision_score(y_train_5, y_train_pred)	# 정밀도
  recall_score(y_train_5, y_train_pred)		# 재현율
  ```

  - F1 점수: 정밀도와 재현율을 하나의 숫자로 만듦(조화평균)

    ![image-20210207232338324](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\image-20210207232338324.png)

    ```python
    from sklearn.metrics import f1_score
    
    f1_score(y_train_5, y_train_pred)
    ```

    - 정밀도와 재현율이 비슷한 분류기는 F1 점수가 높음
    - 상황에 따라 중요한 부분이 달라질 수 있음
    - 정밀도와 재현율은 반비례관계(**정밀도/재현율 트레이트오프**)
    - 정밀도와 재현율이 한쪽으로 너무 치우치는걸 방지하기 위해서(둘 모두의 특성을 반영하기 위해) 조화평균 사용



### 3.3.4 정밀도/재현율 트레이드오프

- 결정함수: SGDClassifier가 분류를 하기 위해 샘플의 점수를 계산하는 함수
- 점수가 임곗값보다 크면 양성클래스, 아니면 음성클래스
- **결정 임곗값**이 가운데 화살표라고 가정
- 오른쪽부분에는 제대로 판별되지 않은 수 하나 존재(정밀도 80%)
- 실제 숫자 5는 6개고 분류기는 4개만 감지(재현율 67%)
- 임계값을 높일 경우 정밀도 향상, 재현율 하락

![image-20210208003557638](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\image-20210208003557638.png)

- 예측에 사용한 점수 확인(사이킷런)

  ```python
  y_scores = sgd_clf.decision_function([some_digit])
  y_scores
  ```

  - 분류기의 `predict()` 메서드 대신 `decision_function()` 메서드 호출: 각 샘플의 점수 도출 가능 이 점수를 기반으로 임곗값 설정 및 예측 가능

  - 임계값이 0일때

    ```python
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)
    ```

  - 임계값을 높혔을 경우

    ```python
    threshold = 8000
    y_some_digit_pred = (y_scores > threshold)
    y_some_digit_pred
    ```

    - 임계값을 높이면 재현율 감소

  - 적절한 임계값 설정

    ```python
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                 method="decision_function")
    ```

    - 예측결과가 아닌 결정점수 반환

    - `precision_recall_curve()` 함수를 사용하여 가능한 모든 임곗값에 대한 정밀도와 재현율 계산

      ```python
      from sklearn.metrics import precision_recall_curve
      
      precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
      ```

    - 맷플롯립을 이용한 임계값 함수 및 정밀도와 재현율 그래프

      ```python
      def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
          plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
          plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
          plt.legend(loc="center right", fontsize=16) # Not shown in the book
          plt.xlabel("Threshold", fontsize=16)        # Not shown
          plt.grid(True)                              # Not shown
          plt.axis([-50000, 50000, 0, 1])             # Not shown
          
      recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
      threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
      
      plt.figure(figsize=(8, 4))                                                                  # Not shown
      plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
      plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
      plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
      plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
      plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
      plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
      save_fig("precision_recall_vs_threshold_plot")                                              # Not shown
      plt.show()
      ```

      - 좋은 정밀도/재현율 트레이드오프를 선택하는 방법: 재현율에 대한 정밀도 곡선을 그리는 것

      ![image-20210208011812393](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\image-20210208011812393.png)

    - 재현율 80%에서 정밀도가 급격히 줄어듦

    - 하강점 직전을 트레이드오프로 선택하는 것이 좋음

  - 정밀도 90% 달성이 목표일 때

    ```python
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    threshold_90_precision
    ```

    - 훈련 세트에 대한 예측

    ```python
    y_train_pred_90 = (y_scores >= threshold_90_precision)
    ```

    - 정밀도 재현율 확인

    ```python
    precision_score(y_train_5, y_train_pred_90)
    recall_score(y_train_5, y_train_pred_90)
    ```



### 3.3.5 ROC 곡선

- 수신기 조작 특성(ROC) 곡선

  - 이진 분류에서 널리 사용하는 도구
  - **거짓 양성 비율(FPR, 양성으로 잘못 분류 된 음성샘플의 비율, 1-진짜음성비율(TNR))**에 대한 **진짜 양성 비율(TPR)** 곡선
  - TNR: 진짜 음성 비율, 특이도
  - 민감도(재현율)에 대한 1-특이도의 그래프
  - `roc_curve()` 함수를 사용하여 여러 임곗값에서 TPR과 FPR 계산

  ```python
  from sklearn.metrics import roc_curve
  
  fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
  ```

  ```python
  # 그래프
  def plot_roc_curve(fpr, tpr, label=None):
      plt.plot(fpr, tpr, linewidth=2, label=label)
      plt.plot([0, 1], [0, 1], 'k--') # 대각 점선
      plt.axis([0, 1, 0, 1])                                    # Not shown in the book
      plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
      plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
      plt.grid(True)                                            # Not shown
  
  plt.figure(figsize=(8, 6))                                    # Not shown
  plot_roc_curve(fpr, tpr)
  fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
  plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
  plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
  plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown
  save_fig("roc_curve_plot")                                    # Not shown
  plt.show()
  ```

  - 여기에도 트레이드오프 존재

  - 재현율이 높을수록 거짓양성이 늘어남

  - **곡선 아래의 면적(AUC)**측정으로 분류기 비교

    ```python
    from sklearn.metrics import roc_auc_score
    
    roc_auc_score(y_train_5, y_scores)
    ```

    - 완벽한 분류기: ROC,AUC=1, 완전 렌덤 분류기: ROC,AUC=0.5

  - 거짓음성보다 거짓양성이 중요할 때는 PR 곡선, 그렇지 않으면 ROC곡선 사용

- `RandomForestClassifier`훈련을 통해 `SGDClassifier`의 ROC곡선과 ROC AUC 점수 비교

  - 훈련 샘플에 대한 점수

    - `RandomForestClassifier`에는 `decision_function()` 메서드가 없음

    - `predict_proba()` 메서드를 통해 샘플이 행, 클래스가 열, 샘플이 주어진 클래스에 속할 확률을 담은 배열 반환

      ```python
      from sklearn.ensemble import RandomForestClassifier
      forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
      y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
      ```
      
- `roc_curve()` 함수는 레이블과 점수 기대
    
  ```python
      y_scores_forest = y_probas_forest[:, 1] # 점수 = 양성 클래스의 확률
      fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
      ```
    
- ROC곡선 그리기
    
  ```bash
      recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]
      
      plt.figure(figsize=(8, 6))
      plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
      plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
      plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
      plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
      plt.plot([fpr_90], [recall_90_precision], "ro")
      plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
      plt.plot([fpr_90], [recall_for_forest], "ro")
      plt.grid(True)
      plt.legend(loc="lower right", fontsize=16)
      save_fig("roc_curve_comparison_plot")
      plt.show()
      ```
    
- 정밀도와 재현율 점수 계산
    
  ```bash
      roc_auc_score(y_train_5, y_scores_forest)
      ```



## 3.4 다중 분류

- 둘 이상의 클래스 구별

- 이진 분류만 가능한 알고리즘도 존재하고 여러 개의 클래스를 직접 처리할 수 있는 알고리즘도 존재

- 이진분류기 여러 개로 다중 클래스 분류 가능

- OvR 전략: 이진분류기 여러 개로 여러 개의 클래스를 만드는 것

- OvO 전략: 각 데이터의 조합으로 이진분류기를 훈련시킴

  - 장점: 전체 훈련세트 중 구별할 두 클래스에 해당하는 샘플만 필요

- 대부분의 이진 분류 알고리즘에서는 OvR 선호

  ```bash
  from sklearn.svm import SVC
  
  svm_clf = SVC(gamma="auto", random_state=42)
  svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train_5이 아니라 y_train입니다
  svm_clf.predict([some_digit])
  ```

  - 5를 구별한 타깃 클래스 대신 원래 타깃 클래스 사용

  - OvO 전략을 사용하여 45개의 이진분류기를 훈련시키고 각각의 결정 점수를 얻어 점수가 가정 높은 클래스 선택

  - 확인

    ```bash
    some_digit_scores = svm_clf.decision_function([some_digit])
    some_digit_scores
    ```

- OvO나 OvR을 강제하려면 `OneVsOneClassifier`나 `OneVsRestClassifier` 사용

  - SVC 기반으로 OvR 전략을 사용하는 다중 분류기 생성

    ```bash
    from sklearn.multiclass import OneVsRestClassifier
    ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
    ovr_clf.fit(X_train[:1000], y_train[:1000])
    ovr_clf.predict([some_digit])
    ```

  - `SGDClassifier` 또는 `RandomForestClassifier` 훈련

    ```bash
    sgd_clf.fit(X_train, y_train)
    sgd_clf.predict([some_digit])
    ```

    - SGD 분류기는 직접 다중 클래스로 분류 가능

      ```bash
      sgd_clf.decision_function([some_digit])
      ```

    - 정확도 평가

      ```bash
      cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
      ```

    - 입력 스케일 조정으로 정확도 향상 가능

      ```bash
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
      cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
      ```



## 3.5 에러 분석

- 실제 프로젝트

  - 데이터 준비 단계에서 가능한 선택 사항 탐색
  - 여러 모델 시도
  - 가장 좋은 모델 중 `GridSearchCV` 사용 하이퍼파라미터 튜닝
  - 자동화

- 에러 종류 분석

  - 오차 행렬

    - `cross_val_predict()`함수로 예측, `confusion_matrix()` 함수 호출

      ```bash
      y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
      conf_mx = confusion_matrix(y_train, y_train_pred)
      conf_mx
      ```

    - 이미지로 표현하여 편하게 관찰

      - 숫자 5가 다른 숫자보다 조금 어두움 -> 분류기가 숫자 5를 다른 숫자만큼 잘 분류하지 못한다는 뜻

    - 오차행렬의 각 값을 대응되는 클래스의 이미지 개수로 나누어 에러 비율을 비교

      ```bash
      row_sums = conf_mx.sum(axis=1, keepdims=True)
      norm_conf_mx = conf_mx / row_sums
      ```

    - 주대각선만 0으로 채워 그래프를 그림

      ```bash
      np.fill_diagonal(norm_conf_mx, 0)
      plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
      save_fig("confusion_matrix_errors_plot", tight_layout=False)
      plt.show()
      ```

    - 분류기가 만든 에러 파악 가능

    - 오차행렬 분석을 통해 분류기 성능 향상 방안에 대한 통찰 탐색 가능

      ```bash
      cl_a, cl_b = 3, 5
      X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
      X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
      X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
      X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
      
      plt.figure(figsize=(8,8))
      plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
      plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
      plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
      plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
      save_fig("error_analysis_digits_plot")
      plt.show()
      ```

    - 선형모델 사용, 선형분류기는 클래스마다 픽셀에 가중치를 할당하고 새로운 이미지에 대해 단순히 픽셀 강도의 가중치 합을 클래스 점수로 계산

    - 이미지를 중앙에 위치시키고 회전되어 있지 않도록 전처리하는 것이 에러를 줄이는 방법

## 3.6 다중 레이블 분류

- 여러 개의 이진 꼬리표를 출력하는 분류 시스템

  ```bash
  from sklearn.neighbors import KNeighborsClassifier
  
  y_train_large = (y_train >= 7)
  y_train_odd = (y_train % 2 == 1)
  y_multilabel = np.c_[y_train_large, y_train_odd]
  
  knn_clf = KNeighborsClassifier()
  knn_clf.fit(X_train, y_multilabel)
  ```

  - 각 숫자 이미지에 두 개의 타깃 레이블이 담긴 `y_mltilabel` 배열을 만들

  - `KNeighborsClassifier` 인스턴스를 만들고 다중 타깃 배열을 사용하여 훈련

    ```bash
    knn_clf.predict([some_digit])
    ```

  -  모든 레이블에 대한 F1 점수의 평균 계산

    ```bash
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    f1_score(y_multilabel, y_train_knn_pred, average="macro")
    ```
    
    - 모든 레이블의 가중치가 같다고 가정
    - 가중치가 다를 경우(레이블에 클래스의 지지도를 가중치로 줬을 경우) `average="weighted"`로 설정

## 3.7 다중 출력 분류

- 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화

- 이미지에서 잡음을 제거하는 시스템

  ```bash
  noise = np.random.randint(0, 100, (len(X_train), 784))
  X_train_mod = X_train + noise
  noise = np.random.randint(0, 100, (len(X_test), 784))
  X_test_mod = X_test + noise
  y_train_mod = X_train
  y_test_mod = X_test
  ```

  - 이미지 훈련 후 깨끗한 이미지

    ```bash
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digit(clean_digit)
    save_fig("cleaned_digit_example_plot")
    ```

    

## 3.8 모르는 부분

### 1. 코드

- `%matplotlib inline`: 그림, 소리, 애니메이션과 같은 Rich output에 대한 표현방식, notebook을 실행한 브라우저에서 바로 그림을 볼 수 있게 해주는 것
- `reshape`: 배열과 차원을 변경해 줌
- [`matplotlib`와 `matplotlib.pyplot` 차이점](https://kongdols-room.tistory.com/72)
  - `matplotlib`: 전체를 아우르는 패키지
  - `matplotlib.pyplot`: `matplotlib`에서 지원하는 모듈 중 하나, 사용환경 인터페이스 제공, 자동으로 figure와 axes 생성, 코드 몇 줄로 그래프 생성 가능, 그래프를 그리기 위해 사용자가 보이지 않는 곳에서 명령을 받아 그래프 작성
- `SGDClassifier`의 매개변수
  - `max_iter`: 계산에 사용할 작업 수
  - `tol`: 정밀도
- `cross_val_score()`: 데이터 프레임과 타겟을 취하고 k-폴드로 분할하고 폴드 평가까지 함
- `numpy.argmax`: 다차원 배열의 경우 차원에 따라 가장 큰 값의 인덱스를 반환



### 2. 내용

- [확률적 경사 하강법](https://everyday-deeplearning.tistory.com/entry/SGD-Stochastic-Gradient-Descent-%ED%99%95%EB%A5%A0%EC%A0%81-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95)

  - 경사하강법 

    - 손실을 줄이는 알고리즘, 미분 값(기울기)가 최소가 되는 점을 찾아 알맞은 가중치 매개변수를 찾아내는 방법

    - 손실함수

      ![캡처](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\캡처.PNG)

      ![캡처1](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\캡처1.PNG)

    1. w1 에 대한 시작점 선택
    2. 시작점에서 손실 곡선의 기울기 계산
       - 단일 가중치에 대한 손실의 기울기는 미분값과 같음
       - 다음 지점을 결정하기 위해 단일 가중의 일부를 시작점에 더함
       - 기울기의 보폭을 통해 손실 곡선의 다음 지점으로 이동
    3. 위의 과정을 반복하며 최소값에 점점 접근

    - 배치가 전체 데이터 셋이라고 가정
    - 배치가 커질 경우 단일 반복으로 계산하는 데에 오랜 시간일 걸림
    - 배치가 커질 경우 중복의 가능성이 높아짐

  - 확률적 경사 하강법

    - 경사하강법보다 적은 계산으로 적절한 기울기를 얻을 수 있음

    - 배치 크기가 1인 경사하강법 알고리즘

    - 데이터 세트에서 무작위로 균일하게 선택한(노이즈가 생길 수 있음) 하나의 예를 의존하여 각 단계의 예측 경사를 계산

      ![캡처2](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\03_classification\캡처2.PNG)

    - 배치: 단일 반복에서 기울기를 계산하는 데 사용하는 예의 총 개수

    - **확룰적**이라는 용어가 각 배치를 포함하는 하나의 예가 무작위로 선택된다는 것을 의미

    - 단점

      - 반복을 충분히 할 경우 효과가 있지만 노이즈가 심함
      - 여러 변형 함수의 최저점에 가까운 점을 찾을 가능성이 높지만 보장x

    - 단점 극복: 미니 배치 SGD

- 손실함수: 예측값과 실제값에 대한 오차

- 정밀도 90%에 대한 임계값이 왜 8000..? 그래프 이상한데

- 입력 스케일이 커질수록 정확도가 올라가는 이유

- 주대각선을 0으로 채워서 에러를 확인하는 이유



### 3. 연구해볼 내용

- 데이터셋 처리 방법
- class 활용방법(python)
- k-fold 방법, 매개변수 cv의 역할
- 곡선 아래의 면적(AUC)을 측정하여 분류기를 비교할 수 있는 이유
- OvO와 OvR 전략의 활용(사이킷런이 알아서 해주기도 함): 훈련세트 크기에 따름