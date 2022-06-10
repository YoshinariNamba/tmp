
# ma, ar ------------------------------------------------------------------
library(forecast)

x1_ma <- arima.sim(model = list(ma = 1), n = 200)
plot(x1_ma)

x2_ma <- arima.sim(model = list(ma = -1), n = 200)
plot(x2_ma)

x1_ar <- arima.sim(model = list(ar = 0.5), n = 200)
plot(x1_ar)

x2_ar <- arima.sim(model = list(ar = -0.5), n = 200)
plot(x2_ar)


# introduction to var -----------------------------------------------------

library(vars)
library(tsDyn)

# e: employment; prod: productivity; rw: real wage; U: unemployment
data(Canada)

# select lag
VARselect(Canada, lag.max = 4)

# estimation
Canada.var <- VAR(y = Canada, p = VARselect(Canada, lag.max = 4)$selection[1])
summary(Canada.var)

# visualize
plot(Canada.var)

# forecast
Canada.pred <- predict(object = Canada.var, n.ahead = 20, ci = 0.95)
plot(Canada.pred)

# simulation
B1 <- matrix(c(0.4, 0.1, 0.2, 0.3), 2)
x <- VAR.sim(B = B1, n = 100, include = "none") # error
x.p <- VARselect(x, lag.max = 10, type = "none")$selection[1]
x.var <- VAR(x, p = x.p, type = "none")

summary(x.var)
plot(x.var)

