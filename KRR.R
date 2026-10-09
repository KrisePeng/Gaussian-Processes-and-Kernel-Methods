library(ggplot2)

# Logistic function used for the simulation
expit <- function(x) 1 / (1 + exp(-x))

# RBF kernel: K(x, x') = exp(-||x-x'||^2 / rho^2)
rbf_kernel <- function(X1, X2 = X1, rho) {
  if (rho <= 0) stop("rho must be positive.")
  X1 <- as.matrix(X1)
  X2 <- as.matrix(X2)
  if (ncol(X1) != ncol(X2)) stop("X1 and X2 need the same number of columns.")
  
  # Squared Euclidean distances between rows of X1 and X2
  sq_dist <- outer(rowSums(X1^2), rowSums(X2^2), "+") -
    2 * X1 %*% t(X2)
  sq_dist <- pmax(sq_dist, 0)  # remove small negative rounding errors
  exp(-sq_dist / rho^2)
}

# Fit KRR and predict at X_new
# Uses the original exercise convention: K + n * lambda * I
KernelRidgeReg <- function(X, y, X_new, rho, lambda) {
  X <- as.matrix(X)
  X_new <- as.matrix(X_new)
  y <- matrix(y, ncol = 1)
  if (nrow(X) != nrow(y)) stop("X and y must have the same number of rows.")
  if (lambda <= 0) stop("lambda must be positive.")
  
  n <- nrow(X)
  K_XX <- rbf_kernel(X, X, rho)
  K_new_X <- rbf_kernel(X_new, X, rho)
  K_regularized <- K_XX + n * lambda * diag(n)
  
  # Solve linear system without explicitly calculating matrix inverse
  alpha <- solve(K_regularized, y)
  y_hat <- K_new_X %*% alpha
  weight <- t(solve(K_regularized, t(K_new_X)))
  list(y.hat = y_hat, weight = weight)
}

# Exploratory nonparametric bootstrap for the mean fitted value
# This is not a validated confidence interval for population E[Y].
KRR_Bootstrap <- function(df, rho, lambda, B = 1000, seed = 42) {
  set.seed(seed)
  n <- nrow(df)
  estimates <- numeric(B)
  for (b in seq_len(B)) {
    index <- sample(seq_len(n), size = n, replace = TRUE)
    df_b <- df[index, , drop = FALSE]
    fit <- KernelRidgeReg(matrix(df_b$X), df_b$Y,
                          matrix(df_b$X), rho, lambda)
    estimates[b] <- mean(fit$y.hat)
  }
  estimates
}

# 1. Simulate nonlinear data
set.seed(42)
n <- 50
X <- runif(n, 0, 10)
Y <- 1 + 2 * expit(2 * X - 10) + rnorm(n, 0, 1)
df <- data.frame(X = X, Y = Y)

# 2. Fit KRR and predict at the observed X > 5 and along a grid
rho <- 1
lambda <- 0.001
X_new <- matrix(X[X > 5], ncol = 1)
fit_subset <- KernelRidgeReg(matrix(X), Y, X_new, rho, lambda)
print(head(fit_subset$y.hat))

X_grid <- seq(0, 10, length.out = 300)
fit_grid <- KernelRidgeReg(matrix(X), Y, matrix(X_grid), rho, lambda)
plot_data <- data.frame(X = X_grid, prediction = as.numeric(fit_grid$y.hat))

p_fit <- ggplot(df, aes(X, Y)) +
  geom_point(alpha = 0.65) +
  geom_line(data = plot_data, aes(X, prediction), linewidth = 0.9) +
  labs(title = "Kernel Ridge Regression with RBF Kernel",
       subtitle = paste("rho =", rho, ", lambda =", lambda),
       x = "X", y = "Y / predicted Y") +
  theme_minimal()
print(p_fit)

# 3. Visualization of the weights for an illustrative prediction
# Each training observation contributes a kernel-based weight to the
# predicted response at the chosen test point.
test_x <- 5
fit_one <- KernelRidgeReg(matrix(X), Y, matrix(test_x, ncol = 1), rho, lambda)
weight_data <- data.frame(X = X, Y = Y,
                          weight = as.numeric(fit_one$weight))
p_weights <- ggplot(weight_data, aes(X, Y, color = weight)) +
  geom_point(size = 2.5) +
  scale_color_gradient(low = "grey80", high = "black") +
  labs(title = paste("Observation weights for prediction at X =", test_x)) +
  theme_minimal()
print(p_weights)

# 4. Compare regularization parameters (lambda)
lambda_values <- c(0.001, 0.1, 10)
lambda_results <- do.call(rbind, lapply(lambda_values, function(lam) {
  fit <- KernelRidgeReg(matrix(X), Y, matrix(X_grid), rho = 1, lambda = lam)
  data.frame(X = X_grid, prediction = as.numeric(fit$y.hat),
             lambda = factor(lam, levels = lambda_values))
}))
p_lambda <- ggplot(lambda_results, aes(X, prediction)) +
  geom_point(data = df, aes(X, Y), inherit.aes = FALSE, alpha = 0.3) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ lambda, labeller = label_both) +
  labs(title = "Effect of KRR Regularization (rho = 1)", y = "Predicted Y") +
  theme_minimal()
print(p_lambda)

# 5. Compare RBF bandwidth parameters (rho)
rho_values <- c(1, 10, 100)
rho_results <- do.call(rbind, lapply(rho_values, function(r) {
  fit <- KernelRidgeReg(matrix(X), Y, matrix(X_grid), rho = r, lambda = 0.001)
  data.frame(X = X_grid, prediction = as.numeric(fit$y.hat),
             rho = factor(r, levels = rho_values))
}))
p_rho <- ggplot(rho_results, aes(X, prediction)) +
  geom_point(data = df, aes(X, Y), inherit.aes = FALSE, alpha = 0.3) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ rho, labeller = label_both) +
  labs(title = "Effect of RBF Bandwidth (lambda = 0.001)", y = "Predicted Y") +
  theme_minimal()
print(p_rho)

# 6. Exploratory bootstrap of average fitted values
bootstrap_results <- KRR_Bootstrap(df, rho = 0.1, lambda = 0.01, B = 1000)
print(c(bootstrap_mean = mean(bootstrap_results),
        bootstrap_sd = sd(bootstrap_results)))
p_boot <- ggplot(data.frame(estimate = bootstrap_results), aes(estimate)) +
  geom_histogram(bins = 30) +
  labs(title = "Bootstrap Distribution of Mean Fitted Values",
       x = "Mean fitted Y", y = "Frequency") +
  theme_minimal()
print(p_boot)

