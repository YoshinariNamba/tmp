
library(tidyverse)

mu <- 0
sigma <- 1
sample_range <- seq(from=-2.5, to=2.5, length.out=1000)
dist <- dnorm(sample_range, mu, sigma)
cdf <- pnorm(sample_range, mu, sigma)

df <- data.frame(x = sample_range, fx = dist, Fx = cdf)

df %>% 
  ggplot(aes(x = x, y = Fx)) + 
  geom_line(color = "blue") +
  geom_vline(xintercept = 0, color = "gray", linetype = "dashed") + 
  theme_classic()


p <- ggplot(data = df, aes(x = x, y = Fx)) + 
  geom_line(color = "#00BFC4") 
p_sm <- p
n_max <- 100
for(i in 2:n_max){
  sigma_sm <- 1/i
  dist_sm <- dnorm(sample_range, mu, sigma_sm)
  cdf_sm <- pnorm(sample_range, mu, sigma_sm)
  
  df_sm <- data.frame(x = sample_range, fx = dist_sm, Fx = cdf_sm)
  if(i < n_max){
    p_sm <- p_sm + geom_line(data = df_sm, aes(x = x, y = Fx), color = "lightgray")
  } else {
    p_sm <- p_sm + geom_line(data = df_sm, aes(x = x, y = Fx), color = "magenta")
  }
}
p_sm + 
  geom_vline(xintercept = 0, color = "gray", linetype = "dashed") + 
  theme_classic()
  
  
  
