---
title: "10장 기계학습 알고리즘1"
author: "jihyun"
date: '2019 8 26 '
output:
  html_document:
    df_print: paged
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. 로지스틱 회기모델

아이리스 데이터로부터 로지스틱 회기 모델을 작성해 보자

```{r cars}
data(iris)
head(iris)
```

sepal.length, sepal.width, petal.length, petal.width 로 species를 예측하는 모델을 만들어 보자.

로지스틱 회기 모델은 두 종류의 예측값만 구별이 가능하므로, virginica, versicolor 두 종류만 남긴다. 

```{r pressure, echo=FALSE}
d <- subset(iris, Species == "virginica" | Species == "versicolor")
str(d)
```

Species 의 factor종류가 3개 이므로 2개로 수정해준다.
```{r}
d$Species <- factor(d$Species)
str(d)
```

모델을 만들어 보자. 

```{r}
(m <- glm(Species ~ ., data = d, family = 'binomial'))
```

모델dl 적합된 값은 fittied()를 통해 알 수 있다.
```{R}
fitted(m)[c(1:5, 51:55)]
```

로지스틱 회기 모델은 0 또는 1로 예측을 하는데 위 결과를 보면 virginica에 해당하는 1:5행은 0, versicolor에 해당하는 51:55 행은 1로 잘 예측된 것을 알 수 있다.

train데이터 중 일부를 test데이터 처럼 사용해 보자. predict()를 통해 모델에 예측을 시킬 수 있다.
type 을 response로 지정하고 예측을 수행하면 0~1 사이의 확률을 구해준다.
```{r}
predict(m, newdata = d[c(1,10,55),], type = "response")
```

앞의 두개와는 근사값이 0, 뒤의 55번째는 근사값이 1로 두 종이 다른것을 잘 구별했다고 할 수 있다.


## 2. 다항 로지스틱 회기 분석

위의 로지스틱 회기 분석은 yes/no의 대답만 할 수있다. 하지만 이를 확장하면 다항 로지스틱 회기 분석을 할 수 있다. 

iris data를 이용해 다항 로지스틱 회기 분석을 해 보자.

```{r}
library(nnet)
(m <- multinom(Species ~ ., data=iris))
```

적성한 모델이 훈련데이터에 어떻게 적합되었는지는 fitted()를 통해 구할 수 있다.
```{r}
head(fitted(m))
```
fitted()의 결과는 각 행의 데이터가 각 분류에 속할 확률을 말한다.
위의 결과로 보면 첫 6개의 행은 전부 setosa종인 확률이 높은 것으로 보인다.

다음은 predict()로 각 종의 데이터 한 행씩을 뽑아 예측을 하게 해보자
```{r}
  predict(m, newdata = iris[c(1,51,101),],type = 'class')
```

type에 probs를 두면 확률을 알 수 있다.

```{r}
  predict(m, newdata = iris[c(1,51,101),],type = 'probs')
```

모델의 정확도를 구해보자. 정확도는 예측값과 실제값의 차이를 통해 알 수 있다.

```{r}
predicted <- predict(m, newdata = iris)
sum(predicted == iris$Species)/NROW(predicted)
```

세부적인 표는 xtabs를 통해 볼 수있다.
```{r}
xtabs(~predicted + iris$Species)
```

## 3. 의사결정나무

의사결정나무는 지니 불순도(Gini Impurity), 또는 정보이득(Information Gain)등의 기준을 사용하여 노드를 재귀적으로 분할하면서 나무 모델을 만드는 방법이다.
의사결정나무는 if-else와 같은 조건문 형태라서 이해하기 쉽고, 속도가 빠르며, 특히 나무 모델 중 random forest는 꽤 괜찮은 성능을 보여주어 캐글에서도 많이 쓰인다고 알려져 있다.

의사결정나무를 이해하는데 도움이 되는 용어들

- 루트노드 : 최상단에 위치한 노드
- 리프노드 : 더이상 자식 노드가 없는 노드
- 불순도 : 지니 불순도를 이용하여 주로 계산한다(지니 불순도는 f(p) = p(1-p)로 정의 되며, 노드에 서로 다른 특징들이 절반씩 있을때 가장 높아지는 그래프 형태이다.). 불순도를 생각했을 때 노드를 나누는 가장 좋은 질문은 자식 노드들의 불순도가 가장 낮아지는 질문이다.


### 가장 일반적인 의사결정나무(rpart 패키지 이용)

아이리스 데이터에 대해 rpart를 이용해 의사결정 나무를 작성해 보자.

```{r}
#install.packages("rpart")
library(rpart)
(m<-rpart(Species ~ ., data = iris,))
```

위의 내용을 plot(), rpart.plot() 함수를 이용하면 좀 더 쉽게 볼 수 있다.

```{r}
#install.packages("rpart.plot")
library(rpart.plot)
prp(m, type = 4, extra=2, digits =3)
```

rpart 또한 predict()를 통해 예측을 어떻게 했는지 볼 수 있다.

```{r}
head(predict(m, newdata = iris, type = 'class'))
```

의사결정나무의 성능을 높이는 방법으로는 prunnung, rpart.control 등의 성능튜닝 방법이 있다.

### 조건부 추론 나무
조건부 추론 나무는 CART(rpart가 구현한 의사 결정 나무 알고리즘) 같은 의사결정나무의 두가지 문제를 해결한다.
1. 통계적유의성 없이 노드룰 분할하는 데에 따른 과적합
2. 다양한 값으로 분할 가능한 변수가 다른 변수들에 비해 선호되는 문제

조건부 추로 ㄴ나무는 조건부 분포에 따라 노드 분할에 사용할 변수를 선택한다.
또한 다중가설검정을 고려한 절차를 이용해서 적당한 시기에 노드의 분활을 중지한다.

아이리스 데이터에 ctree()fmf 적용하여 species 예측 모델을 만들어 보자.

```{r}
#install.packages('party')
library(party)
(m <- ctree(Species ~ ., data = iris))
```

plot()을 통해 만들어진 모델을 보기쉽게 나타낼 수 있다.
```{r}
plot(m)
# plot의 그래프에 칸이 좁아서 setosa밖에보이지 않는 문제는 levles(iris$Species)를 통해 생략된 정보를 알 수 있다.
```

### 랜덤 포레스트(random forest)

랜덤포레스트는 앙상블 학습 기법을 이용한 모델이다.
앙상블 학습은 주어진 데이터로부터 여러개의 모델을 학습한 다음, 예측 시 여러 모델의 예측 결과를 종합해서 정확도를 높히는 방법을 말한다.

랜덤포레스트는 두가지 방법을 이용해서 다양한 의사결정나무를 만든다.
1. 데이터의 일부를 복원추출로 꺼내서 해당 데이터에 대해서만 의사결정나무 만든다.
2. 일부 변수만 대상으로 하여 가지를 나눌 기준을 찾는다.

새로운 데이터에 대한 예측을 할 때는 여러 의사결정나무들이 내놓은 결과를 통해 voting하는 방식으로 최종 결정을 내린다.

랜덤포레스트는 여러개의 나무를 만드는 것으로 과적합 문제를 피한다.

아이리스 데이터에 랜덤 포레스트 모델을 만들어보자.

```{R}
#install.packages("randomForest")
library(randomForest)
(m <- randomForest(Species ~ ., data = iris))
```

OOB : out of bag 의 줄임말으로, 모델 훈련에 사용하지 않은 데이터를 사용한 에러의 추정치이다.

예측에는 predict 함수를 사용한다.

```{r}
head(predict(m, newdata = iris))
```

#### 빠른 모델링을 위한 x y 의직접 지정.

랜덤 포레스트는 500 개의 의사결정나무를 만든다. 따라서 모델링에 걸리는 시간이 길고 데이터가 많아지면 더 오래 걸리게 된다.

포뮬러를 사용하는 방식보다, 설명변수, 종속변수를 직접 지정하면 속도를 더 빠르게 할 수있다.

> m<- randomForest(iris[,1:4], iris[:5])

#### 변수 중요도 평가
변수의 중요도를 평가 할 때에도 ramdomforest를 사용 할 수 있다.
이렇게 구한 변수 중요도는 다른 모델(선형 회기) 에 사용할 변수를 선택 하는데 사용 될 수 있다.

변수의 중요도를 알아보려면 모델 작성시  importance = TRUE 를 지정하면 된다.
그런다음 importance(), varImpPlot() 을 통해 결과를 출력한다.

```{r}
m <- randomForest(Species ~., data = iris, importance = TRUE)
importance(m)
```

정확도 에서는 (Mean Decrease Accuracy)
 petal.length > petal.width > sepal.length > sepal.width
순으로 중요한 것을 알 수 있다.

노트 불순도 계선 (Mean Decrease Gini)
 petal.width > petal.length > sepal.width > sepal.length
순으로 중요한 변수라고 볼 수있다.

varImpPlot() 을 이용하면 좀더 쉽게 볼 수 있다.

```{R}
varImpPlot(m, main = "varImpPlot of iris")
```

### 파라미터 튜닝
  ramdomForest()에는 나무 개수(ntree), 자식 노드로 나눌 때 고려할 변수의 개수(mtry) 등의 파라미터가 있다 기본값으로도 잘 되지만, 성능을 높히기 위해서는 교차 검증을 사용하여 성능을 개선 할 수 있다.
  
  expand.grid()를 이용하여 파라미터의 조합을 만든 후 성능을 차례로 볼 수 있다.

예시로 ntree 를 10, 100, 200 으로, mtry를 3,4 로 바꿔가며 조합해보자

```{r}
(grid <- expand.grid(ntree = c(10, 100, 200), mtry = c(3,4)))
```

파라미터 조합을 10개로 분할한 데이터에 적용하여 모델의 성능을 평가하는 일을 3회 반복하여 최선의 파라미터를 찾아보자
```{r}
#install.packages('cvTools')
#install.packages('foreach')
library(cvTools)
library(foreach)
library(randomForest)
set.seed(719)

K = 10
R = 3
cv <- cvFolds(NROW(iris), K=K, R=R)
grid <-  expand.grid(ntree = c(10, 100, 200), mtry = c(3,4))

result <- foreach(g = 1:NROW(grid), .combine = rbind) %do% {
  foreach(r = 1:R , .combine = rbind) %do%{
    foreach( k =1:K, .combine=rbind) %do%{
      validation_idx <- cv$subsets[which(cv$which == k), r]
      train <- iris[-validation_idx,]
      validation <- iris[validation_idx,]
      
      # 모델 훈련
      m <- randomForest(Species ~., 
                        data = train,
                        ntree = grid[g, "ntree"],
                        mtry = grid[g, "mtry"])
      #예측
      predicted <- predict(m, newdata = validation)
      
      #성능평가
      precision <- sum(predicted == validation$Species) / NROW(predicted)
      return(data.frame(g=g, precision = precision))
      
    }
  }
}
                     
result
```


g 값 마다 묶어서 평균을 구해보자
```{r}
#install.packages('plyr')
library(plyr)
ddply(result, .(g), summarise, mean_precision = mean(precision))
```

가장 높은 성능은 g=4, g=6일 때다.
이 조합을 grid에서 찾아보면 ntree = 10, mtry = 4인 경우와, ntree = 200, mtry = 4인 경우이다.
```{r}
grid[c(4,6),]
```


### 정규화 랜덤 포레스트(RRF, Regulized Random Forest)
정규화 랜덤 포레스트는 랜덤포레스트의 변수 선택 방식을 개선한 방식으로,
부모 노드에서 가지를 나눌 때 사용한 변수와 유사한 변수로 자식 노드에서 가지를 나눌 경우, 해당 변수에 패널티를 부여한다.

이 방식은은 랜덤 포레스트에 비해 좋은 변수를 선택하는 것으로 알려져 있으며
RRF 패키지를 통해 이용 할 수 있다.

