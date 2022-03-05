
# library
library(tidyverse)

# seed random number
set.seed(2022)

# sample size
n <- 1000

# parameter
mu <- c(2, 2, 0)
Sigma <- matrix(c(1, 0.8, 0.2, 0.8, 1, 0, 0.2, 0, 1), 
                nrow = 3, ncol = 3, byrow = TRUE)
beta <- 1

# variable
endo_var <- MASS::mvrnorm(n, mu, Sigma)
x <- endo_var[, 1]
z <- endo_var[, 2]
u <- endo_var[, 3]
y <- beta*x + u
x_hat <- lm(x ~ z) %>% fitted()

# pack
df <- data.frame(y, x, x_hat, z)

# plot
df %>% 
  ggplot(aes(x = x, y = y)) + 
  geom_point() +
  geom_smooth(aes(x = x, y = y, color = "ols"), 
              method = 'lm', lty = 'dashed', se = F) +
  geom_smooth(aes(x = x_hat, y = y, color = 'IV'), 
              method = 'lm', lty = 'dashed', se = F) +
  geom_abline(color = 'green', 
              intercept = 0, slope = beta) + 
  scale_color_manual(name = "estimator", values = c('orange', 'blue'))

# 3d
plt1 <- 
  df %>% 
  ggplot(aes(x = z, y = x)) +
  geom_point()

plt2 <- 
  df %>% 
  ggplot(aes(x = y, y = x)) +
  geom_point()

plt3 <- 
  df %>% 
  ggplot(aes(x = z, y = y)) +
  geom_point()

gridExtra::grid.arrange(plt1, plt2, plt3, nrow = 2)
