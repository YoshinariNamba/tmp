

# 1. preparation -------------------------------------------------------------

# 1. 1 Setup R ############################################################
# initialize
rm(list = ls())

# library
pacman::p_load(
  tidyverse, 
  skimr, 
  evd, 
  numDeriv, 
  #plot3D
)


# 1. 2 Data Generation #####################################################
# set parameter
## distance and price
theta_true <- c(theta_c = 0.004, theta_p = 0.003)

## discount factor
beta <- 0.99

## euler's constant
Euler_const <- digamma(1)

## number of choice
num_choice <- 2


# define state variable
## price
price_states <- seq(2000, 2500, by = 100)

## distance
mileage_states <- seq(0, 100, by = 5)

## # of price
num_price_states <- length(price_states)

## # of distance
num_mileage_states <- length(mileage_states)

## # of state variables
num_states <- num_price_states*num_mileage_states

## express the combination as a data frame
state_df <- 
  tibble(
    state_id = 1:num_states, 
    price_id = rep(1:num_price_states, times = num_mileage_states), 
    mileage_id = rep(1:num_mileage_states, each = num_price_states), 
    price = rep(price_states, times = num_mileage_states), 
    mileage = rep(mileage_states, each = num_price_states)
  )

state_df %>% 
  tail(3)


# transition matrix for mileage
gen_mileage_trans <- function(kappa){
  # define prob for move to upper bins
  kappa_1 <- kappa[1] 
  kappa_2 <- kappa[2]
  
  # if not buy a new car
  ## empty transition matrix
  mileage_trans_mat_hat_not_buy <- 
    matrix(0, ncol = num_mileage_states, nrow = num_mileage_states)
  
  ## substitute probabilities
  for(i in 1:num_mileage_states){
    for(j in 1:num_mileage_states){
      if(i == j){
        mileage_trans_mat_hat_not_buy[i, j] <- 1 - kappa_1 - kappa_2
      } else if(i == j -1){
        mileage_trans_mat_hat_not_buy[i, j] <- kappa_1
      } else if(i == j - 2){
        mileage_trans_mat_hat_not_buy[i, j] <- kappa_2
      }
    }
  }
  
  ### special case
  mileage_trans_mat_hat_not_buy[num_mileage_states - 1, num_mileage_states] <- 
    kappa_1 + kappa_2
  mileage_trans_mat_hat_not_buy[num_mileage_states, num_mileage_states] <- 1
  
  # if buy a new car
  mileage_trans_mat_hat_buy <- 
    matrix(1, nrow = num_mileage_states, ncol = 1) %*%
    mileage_trans_mat_hat_not_buy[1, ] # "%*%" represents matrix product
  
  return(
    array(
      c(mileage_trans_mat_hat_not_buy, mileage_trans_mat_hat_buy), 
      dim = c(num_mileage_states, num_mileage_states, num_choice)
    )
  )
}

# transition matrix for price
gen_price_trans <- function(lambda){
  lambda_11 <- 1 - lambda[1] - lambda[2] - lambda[3] - lambda[4] - lambda[5]
  lambda_22 <- 1 - lambda[6] - lambda[7] - lambda[8] - lambda[9] - lambda[10]
  lambda_33 <- 1 - lambda[11] - lambda[12] - lambda[13] - lambda[14] - lambda[15]
  lambda_44 <- 1 - lambda[16] - lambda[17] - lambda[18] - lambda[19] - lambda[20]
  lambda_55 <- 1 - lambda[21] - lambda[22] - lambda[23] - lambda[24] - lambda[25]
  lambda_66 <- 1 - lambda[26] - lambda[27] - lambda[28] - lambda[29] - lambda[30]
  price_trans_mat_hat <- 
    c(lambda_11, lambda[1], lambda[2], lambda[3], lambda[4], lambda[5],
      lambda[6], lambda_22, lambda[7], lambda[8], lambda[9], lambda[10],
      lambda[11], lambda[12], lambda_33, lambda[13], lambda[14], lambda[15],
      lambda[16], lambda[17], lambda[18], lambda_44, lambda[19], lambda[20],
      lambda[21], lambda[22], lambda[23], lambda[24], lambda_55, lambda[25],
      lambda[26], lambda[27], lambda[28], lambda[29], lambda[30], lambda_66) %>% 
    matrix(ncol = num_price_states, nrow = num_price_states, byrow=T)
  return(price_trans_mat_hat)
}

# set probabilities for mileage matrix
kappa_true <- c(0.25, 0.05)

mileage_trans_mat_true <- gen_mileage_trans(kappa_true)

# view generated trans matrix for mileage
mileage_trans_mat_true[1:4, 1:4, 1]


# set probabilities for price trans
lambda_true <- c(0.1, 0.2, 0.2, 0.2, 0.2,
                 0.1, 0.2, 0.2, 0.2, 0.2,
                 0.1, 0.1, 0.2, 0.2, 0.1,
                 0.1, 0.1, 0.2, 0.2, 0.1,
                 0.05, 0.05, 0.1, 0.1, 0.2,
                 0.05, 0.05, 0.1, 0.1, 0.2)

price_trans_mat_true <- gen_price_trans(lambda_true)

# view generated trans matrix for price
price_trans_mat_true

# transition matrix per control variable
trans_mat_true <- list()

# #control variable 1: not_buy
trans_mat_true$not_buy <- 
  mileage_trans_mat_true[, , 1] %x% price_trans_mat_true # "%x%" represents kronecker product

## control variable 2: buy
trans_mat_true$buy <- 
  mileage_trans_mat_true[, , 2] %x% price_trans_mat_true


# calculate steady state
## solve the following
## price_dist_steady %*%  price_trans_mat == price_dist_steady
price_trans_eigen <- eigen(t(price_trans_mat_true))

## steady distribution of price
price_dist_steady <- 
  price_trans_eigen$vectors[, 1]/sum(price_trans_eigen$vectors[, 1])

price_dist_steady




