# 6장 결정 트리

- 결정트리: 분류와 회귀 작업, 다중출력 작업도 가능한 머신러닝 알고리즘 (복잡한 데이터셋 학습 가능)



## 6.1 결정 트리 학습과 시각화

- 붓꽃 데이터 셋 `DecisionTreeClassifier` 훈련 코드

  ```bash
  from sklearn.datasets import load_iris
  from sklearn.tree import DecisionTreeClassifier
  
  iris = load_iris()
  X = iris.data[:, 2:] # 꽃잎 길이와 너비
  y = iris.target
  
  tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
  tree_clf.fit(X, y)
  ```

- `export_graphviz()` 함수를 이용해 그래프 정의를 `iris_tree.dot` 파일로 출력하여 훈련된 결정 트리 시각화

  ```bash
  from graphviz import Source
  from sklearn.tree import export_graphviz
  
  export_graphviz(
          tree_clf,
          out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
          feature_names=iris.feature_names[2:],
          class_names=iris.target_names,
          rounded=True,
          filled=True
      )
  
  Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
  ```

  ![image-20210223013352566](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210223013352566.png)



## 6.2 예측하기

- 예측을 만들어내는 순서 확인

- 새로 발견한 붓꽃의 품종을 분류하려 한다고 가정

- **루트 노드**(깊이가 0인 맨 꼭대기의 노드)에서 시작

  - 꽃잎의 길이가 2.45cm보다 짧은지 검사
    - true -> 왼쪽의 자식노드로 이동 -> **리프노드**(자식노드를 가지지 않는 노드)이므로 pass -> Iris-Setosa라고 예측
    - false -> 오른쪽의 자식노드로 이동 -> 꽃잎의 너비가 1.75cm보다 작은지 검사
      - true -> Iris-Vericolor라고 예측
      - false -> Iris-Virginica라고 예측

- 노드의 `sample`속성은 얼마나 많은 훈련 샘플이 적용되었는지 해아린 것

  - 100개의 훈련 샘플의 꽃잎 길이가 2.45cm보다 길다
  - 그 중 54개의 꽃잎의 너비가 1.75cm보다 짧다

- 노드의 `value` 속성은 노드에서 각 클래스에 얼마나 많은 훈련 샘플이 있는지 알려줌

  - 맨 오른쪽 아래 노드는 Iris-Setosa가 0개, Iris-Versicolor가 1개, Iris-Virginica가 45개

- 노드의 `gini`속성은 **불순도** 측정

  - 한 노드의 모든 샘플이 같은 클래스에 속해있다면 이 노드는 순수하다고 표현 (`gini=0`)

  - 지니불순도

    ![image-20210223015928834](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210223015928834.png)

    - 이 식에서 *pik*는 i번째 노드에 있는 훈련 샘플 중 클래스 k에 속한 샘플의 비율

- ![image-20210223020114213](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210223020114213.png)

  - 굵은 수직선이 루트 노드(깊이 0)의 결정 경계 (꽃잎 길이 = 2.45cm)를 보여줌
  - 왼쪽 영역은 순수노드만 있기 떄문에 나눌 수 없음
  - 오른쪽 영역은 순수노드가 아니기 때문에 깊이 1의 오른쪽 노드는 꽃잎 너비 1.75cm에서 나누어짐(파선)
  - `max_depth`를 2로 설정했기 때문에 결정트리는 여기에서 분할 끝, `max_depth`를 3으로 설정하면 깊이 2의 두 노드가 각각 결정 경계를 추가로 만듦(점선)

- 모델 해석: 화이트박스와 블랙박스

  - 결정트리는 직관적이고 결정 방식을 이해하기 쉬움(**화이트박스**)
  - 렌덤 포레스트, 신경망(**블랙박스**)
  - 블랙박스: 성능이 뛰어나고, 예측을 만드는 연산 과정을 쉽게 확인할 수 있음. 하지만 과정을 쉽게 파악하는것이 어려움



## 6.3 클래스 확률 추정

- 한 샘플이 특정 클래스 *k*에 속할 확률을 추정하는 것 가능
- 이 샘플에 대해 리프노드를 찾기 위해 트리를 탐색
- 그 노드에 있는 클래스 *k*의 훈련 샘플의 비율 반환

- 길이가 5cm이고, 너비가 1.5cm인 꽃잎 발견 가정

  - 리프노드: 깊이 2에서 왼쪽 노드
  - 결정트리: 그에 해당하는 확률 출력
  - Iris-Setosa(0%,0/54), Iris-Versicolor(90.7%,49/54), Iris-Virginica(9.3%,5/54)
  - 이 확률을 통해 Iris-Versicolor 출력

  ```bash
  tree_clf.predict_proba([[5, 1.5]])
  tree_clf.predict([[5, 1.5]])
  ```



## 6.4 CART 훈련 알고리즘

- 사이킷런은 결정트리를 훈련시키기 위해 CART 알고리즘 사용

  ![image-20210223022013329](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210223022013329.png)

  - CART 알고리즘을 통해 훈련 세트를 둘로 나눔, 계속해서 서브셋으로 나눔
  - 이 과정은 최대 깊이가 되면 중지하거나 불순도를 줄이는 분할을 찾을 수 없을 때 중지



## 6.5 계산 복잡도

- 예측을 하려면 결정트리를 루트 노드에서부터 리프 노드까지 탐색
- 일반적으로 결정트리는 거의 균형을 이루고 있으며, 결정트리를 탐색하기 위해서는 약 ***O(log(m))*** 개의 노드를 거쳐야 함 (노드 수 고정)
- 훈련 알고리즘은 각 노드에서 모든 훈련 샘플의 모든 특성을 비교
- 각 노드에서 모든 샘플의 모든 특성을 비교하면 훈련 복잡도가 ***O(nxmlog(m))***이 됨

- 훈련 세트가 작을 경우, 데이터 정렬을 통해 훈련 속도 높힘



## 6.6 지니 불순도 또는 엔트로피?

- 기본적으로는 지니 불순도 사용, `criterion` 매개변수를 `"entropy"`로 지정하여 **엔트로피** 불순도 사용 가능

- 엔트로피: 분자의 무질서함을 측정, 원래는 열역학 개념, 자료가 비슷할 경우 엔트로피 지수가 낮아짐

- 머신러닝에서 불순도 측정 방법으로 사용

- 엔트로피

  ![image-20210223024526557](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210223024526557.png)

  - 계산이 더 빠르기 때문에 일반적으로는 지니 불순도 사용
  - 다른 트리가 만들어지는 경우, 엔트로피가 조금 더 균형잡힌 트리를 만듦



## 6.7 규제 매개변수

- 비파라미터 모델
  - 결정트리는 훈련 데이터에 대한 제약 사항이 거의 없음(과대적합되기 쉬움)
  - 훈련되기 전에 파라미터 수가 결정되지 않기 떄문에 **비파라미터 모델**이라고 함
  - 고정 x, 자유로움
- 파라미터 모델
  
  - 미리 정의된 모델 파라미터 수를 가지고 있기 때문에 자유도가 제한되고 과대적합의 위험이 적음(과소적합 가능)
- 사이킷런에서 `max_depth` 매개변수로 모델을 규제하고, 과대적함의 위험을 감소시킴
- `DecisionTreeClassifier` 에 형태를 제한하는 매개변수 존재
  - `min_samples_split`: 분할되기 위해 노드가 가져야 하는 최소 샘플 수
  - `min_samples_leaf`: 리프 노드가 가지고 있어야 할 최소 샘플 수
  - `min_weight_fraction_leaf`: 위와 같지만 가중치가 부여된 전체 샘플 수에서의 비율
  - `max_leaf_nodes`: 리프 노드의 최대 수
  - `max_features`: 각 노드에서 분할에 사용할 특성의 최대 수
  - `min_`으로 시작하는 매개변수 증가, `max_`로 시작하는 매개변수 감소로 모델의 규제가 커짐

- moons 데이터셋에 훈련시킨 두 개의 결정트리

  ![image-20210223030246220](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210223030246220.png)

  - 기본 매개변수 사용(왼쪽)
  - `min_samples_leaf=4`로 지정하여 훈련
  - 왼쪽이 과대적합



## 6.8 회귀

- 사이킷런의 `DecisionTreeRegressor`를 사용하여 잡음이 섞인 2차 함수 형태의 데이터셋에서 `max_depth=2` 설정으로 회귀 트리

  ```bash
  from sklearn.tree import DecisionTreeRegressor
  
  tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
  tree_reg.fit(X, y)
  ```

  ![image-20210224143812832](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210224143812832.png)

  - 클래스를 예측하는 대신 어떤 값을 예측

    ![image-20210224144854220](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210224144854220.png)

  - 각 영역의 예측값은 항상 그 영역에 있는 타깃값의 평균이 됨

- CART 알고리즘은 훈련 세트를 불순도를 최소화하는 방향으로 분할하는 대신 MSE를 최소화하도록 분할

  ![image-20210224145458533](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210224145458533.png)

- 회귀 작업에서도 결정 트리가 과대적합되기 쉬움

- `min_samples_leaf`변수 지정을 통해 규제

  ![image-20210224161421166](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210224161421166.png)



## 6.9 불안정성

- 계단 모양의 결정 경계를 만들기 때문에 훈련 세트의 회전에 민감함

- PCA 기법 사용을 통해 좋은 방향으로 회전

  ![image-20210224161609694](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210224161609694.png)

  - 훈련 데이터의 작은 변화에도 민감

  - 데이터 일부를 지우고 다시 트리를 만들 경우, 다른형태로 변함

    ![image-20210224164420549](C:\Users\sma05\Desktop\machine learning study\hands-on machine learning\06_decision_trees\image-20210224164420549.png)

    

## 6.10 모르는 부분

### 1. 코드

### 2. 내용

### 3. 연구 부분

