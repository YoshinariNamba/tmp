教師あり機械学習 draft
================
Yoshinari Namba
2021/01/04

## 0. 準備

### パッケージ

``` r
## libary
library(tidyverse)
library(tidymodels)
library(glmnet)
library(stargazer)
```

### データ

ボストンの住宅価格データを使う。

``` r
## data source
data("Boston", package = "MASS")
df <- Boston
rm(Boston)
```

目的変数と予測変数を別のデータセットとして格納。予測変数の交差項・2乗項を作成。

``` r
## 予測変数(説明変数)を抽出
X_simple <- df %>% 
  select(-medv)

## 予測変数の交差項と2乗項を新たな予測変数として作成
X <- recipe(~ ., data = X_simple) %>%
  step_interact(~all_predictors():all_predictors()) %>% # Make interactions
  step_poly(crim, zn, indus) %>% # Make 2nd order terms
  step_nzv(all_predictors()) %>%  # Remove nearly zero variance variables
  step_lincomb(all_predictors()) %>% # Remove linear combinations
  prep() %>% 
  bake(X_simple) %>% 
  as.data.frame()

## データフレームに変換
X_simple <- as.data.frame(X_simple)

## 目的変数
y <- df$medv
```

-   観察単位: 町
-   変数 (`?MASS::Boston`参照)  
    – medv: median value of owner-occupied homes in $1000s (目的変数)  
    – crim: per capita crime rate by town  
    – zn: proportion of residential land zone for lots over 25,000
    sq.ft.  
    – indus: proportion of non-retail business acre per town  
    – chas: Charles River dummy variable  
    – nox: nitrogen oxides concentration  
    – rm: average room per dwelling  
    – age: proportion of owner-occupied units built prior to 1940  
    – dis: weighed mean of distances to five Boston employment centers  
    – rad: index of accessibility to radial highways  
    – tax: full-value property-tax rate per $10,000  
    – ptratio: pupil-teacher ratio by town  
    – black:
    ![1000(Bk - 0.63)^2](https://latex.codecogs.com/png.latex?1000%28Bk%20-%200.63%29%5E2 "1000(Bk - 0.63)^2")
    where Bk is the proportion of blacks by town  
    – lstat: lower status of the population (percent)

``` r
df %>% stargazer(type = "text")
```

    ## 
    ## ===============================================================
    ## Statistic  N   Mean   St. Dev.  Min   Pctl(25) Pctl(75)   Max  
    ## ---------------------------------------------------------------
    ## crim      506  3.614   8.602   0.006   0.082    3.677   88.976 
    ## zn        506 11.364   23.322    0       0       12.5     100  
    ## indus     506 11.137   6.860   0.460   5.190    18.100  27.740 
    ## chas      506  0.069   0.254     0       0        0        1   
    ## nox       506  0.555   0.116   0.385   0.449    0.624    0.871 
    ## rm        506  6.285   0.703   3.561   5.886    6.624    8.780 
    ## age       506 68.575   28.149  2.900   45.025   94.075  100.000
    ## dis       506  3.795   2.106   1.130   2.100    5.188   12.126 
    ## rad       506  9.549   8.707     1       4        24      24   
    ## tax       506 408.237 168.537   187     279      666      711  
    ## ptratio   506 18.456   2.165   12.600  17.400   20.200  22.000 
    ## black     506 356.674  91.295  0.320  375.378  396.225  396.900
    ## lstat     506 12.653   7.141   1.730   6.950    16.955  37.970 
    ## medv      506 22.533   9.197     5      17.0      25      50   
    ## ---------------------------------------------------------------

## 1. OLSの実行

### 1-1. 学習

シンプルなモデルのAdjusted R2

``` r
mdl_lm_simple <- lm(data = X_simple, formula = y ~ .)
summary(mdl_lm_simple)$adj.r.squared
```

    ## [1] 0.7337897

複雑なモデルのAdjusted R2 (交差項と2乗項を予測変数に含めたモデル)

``` r
mdl_lm_complex <- lm(data = X, formula = y ~ .)
summary(mdl_lm_complex)$adj.r.squared
```

    ## [1] 0.8875087

### 1-2. 予測誤差 (Mean Squared Error; MSE)

交差項や2乗項を予測変数として含めた複雑なモデルの方が予測誤差(MSE)が小さい．

``` r
# modelling
mse_lm_simple <- sum( ( y - predict(mdl_lm_simple, X_simple) )^2 ) / length(y)
mse_lm_complex <- sum( ( y - predict(mdl_lm_complex, X) )^2 ) / length(y)

# summary
mse_lm <- data.frame(simple = mse_lm_simple, complex = mse_lm_complex)
rownames(mse_lm) <- "MSE"

mse_lm
```

    ##       simple  complex
    ## MSE 21.89483 8.029685

## 2. Penalized Regressionの方法論

### 2-1. Bias-Variance Tradeoff

![y,X](https://latex.codecogs.com/png.latex?y%2CX "y,X")の間に![y=f(X)+\\epsilon](https://latex.codecogs.com/png.latex?y%3Df%28X%29%2B%5Cepsilon "y=f(X)+\epsilon")という関係があると仮定する(![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon")は平均![0](https://latex.codecogs.com/png.latex?0 "0"),
分散![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2")の誤差)．データ![D](https://latex.codecogs.com/png.latex?D "D")から![f(X)](https://latex.codecogs.com/png.latex?f%28X%29 "f(X)")を近似する![\\hat{f}(X; D)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28X%3B%20D%29 "\hat{f}(X; D)")というモデルを構築して新たな![X](https://latex.codecogs.com/png.latex?X "X")から![y](https://latex.codecogs.com/png.latex?y "y")を予測する．このとき予測誤差の期待値はBias(![Bias_D(.)^2](https://latex.codecogs.com/png.latex?Bias_D%28.%29%5E2 "Bias_D(.)^2"))とVariance(![Var\_{D}(.)](https://latex.codecogs.com/png.latex?Var_%7BD%7D%28.%29 "Var_{D}(.)")),
Noise(![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2"))に分解できる．

![
\\begin{align\*}
  E\_{D, \\epsilon }\\left\[(y - \\hat{f}(X ; D))^2 \\right\] &= \\left\[ Bias\_{D} ( \\hat{f} (X ; D) ) \\right\]^2 +  Var\_{D} \\left\[ \\hat{f} (X ; D) \\right\] + \\sigma^2
\\end{align\*}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%2A%7D%0A%20%20E_%7BD%2C%20%5Cepsilon%20%7D%5Cleft%5B%28y%20-%20%5Chat%7Bf%7D%28X%20%3B%20D%29%29%5E2%20%5Cright%5D%20%26%3D%20%5Cleft%5B%20Bias_%7BD%7D%20%28%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%29%20%5Cright%5D%5E2%20%2B%20%20Var_%7BD%7D%20%5Cleft%5B%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%5Cright%5D%20%2B%20%5Csigma%5E2%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
  E_{D, \epsilon }\left[(y - \hat{f}(X ; D))^2 \right] &= \left[ Bias_{D} ( \hat{f} (X ; D) ) \right]^2 +  Var_{D} \left[ \hat{f} (X ; D) \right] + \sigma^2
\end{align*}
")

ここで，

![
\\begin{align\*}
  Bias\_{D} ( \\hat{f} (X ; D) ) 
  &= E\_{D}(\\hat{f}(X;D)) - f(X) \\\\
  Var\_{D} \\left\[ \\hat{f} (X ; D) \\right\]
  &= E\_{D} \\left\[ ( E\_{D}\[\\hat{f}(X; D)\] - \\hat{f}(X; D) )^2 \\right\].
\\end{align\*}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%2A%7D%0A%20%20Bias_%7BD%7D%20%28%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%29%20%0A%20%20%26%3D%20E_%7BD%7D%28%5Chat%7Bf%7D%28X%3BD%29%29%20-%20f%28X%29%20%5C%5C%0A%20%20Var_%7BD%7D%20%5Cleft%5B%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%5Cright%5D%0A%20%20%26%3D%20E_%7BD%7D%20%5Cleft%5B%20%28%20E_%7BD%7D%5B%5Chat%7Bf%7D%28X%3B%20D%29%5D%20-%20%5Chat%7Bf%7D%28X%3B%20D%29%20%29%5E2%20%5Cright%5D.%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
  Bias_{D} ( \hat{f} (X ; D) ) 
  &= E_{D}(\hat{f}(X;D)) - f(X) \\
  Var_{D} \left[ \hat{f} (X ; D) \right]
  &= E_{D} \left[ ( E_{D}[\hat{f}(X; D)] - \hat{f}(X; D) )^2 \right].
\end{align*}
")

学習に使用したデータとは別のinputから予測する場合は，モデルを複雑化するとBiasは縮小する一方でVarianceが拡大するトレードオフの関係がある．

### 2-2. Ridge回帰・Lasso回帰

OLSは「誤差の2乗和」を最小化するのに対し，Ridge回帰
(Lasso回帰)では「誤差にパラメータの2乗(絶対値)を加えた値」を最小化する．パラメータの2乗(絶対値)はモデルの複雑さに対する罰則(penalty)である．

![
\\begin{align\*}
  OLS &: \\min\_{\\beta} \\  (y - X \\beta)^{T}(y - X \\beta) \\\\
  Ridge &: \\min\_{\\beta} \\  (y - X \\beta)^{T}(y - X \\beta) + \\lambda \\beta^{T}\\beta \\\\
  Lasso &: \\min\_{\\beta} \\  (y - X \\beta)^{T}(y - X \\beta) + \\lambda \|\| \\beta \|\|\_1
\\end{align\*}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%2A%7D%0A%20%20OLS%20%26%3A%20%5Cmin_%7B%5Cbeta%7D%20%5C%20%20%28y%20-%20X%20%5Cbeta%29%5E%7BT%7D%28y%20-%20X%20%5Cbeta%29%20%5C%5C%0A%20%20Ridge%20%26%3A%20%5Cmin_%7B%5Cbeta%7D%20%5C%20%20%28y%20-%20X%20%5Cbeta%29%5E%7BT%7D%28y%20-%20X%20%5Cbeta%29%20%2B%20%5Clambda%20%5Cbeta%5E%7BT%7D%5Cbeta%20%5C%5C%0A%20%20Lasso%20%26%3A%20%5Cmin_%7B%5Cbeta%7D%20%5C%20%20%28y%20-%20X%20%5Cbeta%29%5E%7BT%7D%28y%20-%20X%20%5Cbeta%29%20%2B%20%5Clambda%20%7C%7C%20%5Cbeta%20%7C%7C_1%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
  OLS &: \min_{\beta} \  (y - X \beta)^{T}(y - X \beta) \\
  Ridge &: \min_{\beta} \  (y - X \beta)^{T}(y - X \beta) + \lambda \beta^{T}\beta \\
  Lasso &: \min_{\beta} \  (y - X \beta)^{T}(y - X \beta) + \lambda || \beta ||_1
\end{align*}
")

ここでハイパーパラメータ![\\lambda](https://latex.codecogs.com/png.latex?%5Clambda "\lambda")は交差検証(後述)によって予測誤差を最小化する値にチューニングされる．

## 3. 交差検証 (Cross-Validation)

### 3-1. 交差検証による予測誤差の評価

未知のtargetを予測する際，biasのみならずvarianceも縮小させたい．そこで，手元のデータを「学習データ(training
data)」と予測精度を評価する「テストデータ(test
data)」に分割してモデルを吟味する「交差検証」を行う．学習データとテストデータはだいたい7:3ないし8:2に分割することが多い．

``` r
set.seed(2022)
N_train <- round(length(y)*0.7) #  # of training observations 
id_train <- sample(x = length(y), size = N_train, replace = FALSE) # IDs of training observations

# training data
y_train <- y[id_train]
X_train <- X[id_train, ]

# test data
y_test <- y[-id_train]
X_test <- X[-id_train, ]
```

OLS, Ridge回帰, Lasso回帰に基づいてtraining dataからモデルを学習し，test
dataからモデルの予測誤差(MSE)を算出してモデルを評価する．

``` r
## modelling
mdl_lm <- lm(data = X_train, formula = y_train ~.)
mdl_ridge <- cv.glmnet(x = as.matrix(X_train), y = y_train, alpha = 1)
mdl_lasso <- cv.glmnet(x = as.matrix(X_train), y = y_train, alpha = 0)

## test
MSE_lm <- sum( (y_test - predict(mdl_lm, X_test))^2 ) / length(y)
MSE_ridge <- sum( (y_test - as.vector(predict(mdl_ridge, as.matrix(X_test))))^2 ) / length(y)
MSE_lasso <- sum( (y_test - as.vector(predict(mdl_lasso, as.matrix(X_test))))^2 ) / length(y)

## summary
MSE <- data.frame(OLS = MSE_lm, Ridge = MSE_ridge, Lasso = MSE_lasso)
rownames(MSE) <- "MSE"

MSE
```

    ##          OLS    Ridge    Lasso
    ## MSE 7.214426 5.476845 8.195149

#### \* 交差検証によるハイパーパラメータのチューニング

Ridge回帰及びLasso回帰のモデリングではハイパーパラメータ![\\lambda](https://latex.codecogs.com/png.latex?%5Clambda "\lambda")の最適な値を選定するためにtraining
dataの中で交差検証が行われている(Nested
Cross-Validation)．ただし，ハイパーパラメータのチューニングにおける交差検証は10-foldと呼ばれる手法を用いており，予測誤差の評価で用いた手法(holdout)とは異なる．
