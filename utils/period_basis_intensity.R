library(splines2)

generate_lambda0_intensity <- function(
    K = 5,                     # Number of B-spline basis
    m = 3,                     # Number of event types
    total_time = 20,           # Total time range
    period_length = 10,        # Period length for cycles
    seed = 42,                 # Random seed
    lambda0_gamma_range = c(0, 1),    # Range for beta parameters
    lambda0_theta_range = c(0, 0.5),  # Range for theta parameters
    periodic_weight = 1.0)            # Weight of periodic component [0,1]
{ # Generate periodic intensity functions for multiple event types
  
  set.seed(seed)
  
  # Generate parameters
  params <- list(
    K = K,
    m = m,
    total_time = total_time,
    period_length = period_length,
    # Parameters for periodic component
    lambda0_theta = runif(K, lambda0_theta_range[1], lambda0_theta_range[2]),
    lambda0_gamma = matrix(runif(m*K, lambda0_gamma_range[1], lambda0_gamma_range[2]), nrow = m),
    # Parameters for non-periodic component
    lambda0_base = runif(m, 0, 0.24),  # Baseline intensity for non-periodic component
    periodic_weight = periodic_weight
  )
  
  # Create periodic B-spline basis
  pbs_matrix <- bSpline(
    x = seq(0, period_length, length.out = 100),
    df = K,
    degree = 3,
    Boundary.knots = c(0, period_length),
    intercept = TRUE,
    periodic = TRUE
  )
  
  # Create intensity functions
  intensity_funcs <- lapply(1:m, function(i) {
    function(t) {
      # Periodic component
      cyclic_t <- t %% period_length
      basis <- predict(pbs_matrix, cyclic_t)
      periodic_part <- as.numeric(basis %*% (params$lambda0_gamma[i, ] * params$lambda0_theta))
      
      # Non-periodic component
      nonperiodic_part <- params$lambda0_base[i]
      
      # Weighted mixture
      params$periodic_weight * periodic_part + 
        (1 - params$periodic_weight) * nonperiodic_part
    }
  })
  
  list(intensity_funcs = intensity_funcs, params = params)
}

plot_intensity_functions <- function(
    intensity_obj,             # Output from generate_lambda0_intensity
    colors = c("#1f77b4", "#ff7f0e", "#2ca02c"), # Default tableau colors
    plot_title = "Baseline Intensity Functions",
    resolution = 1000) {        # Plotting resolution
    
    # Plot generated intensity functions
  
  # Extract parameters
  t_max <- intensity_obj$params$total_time
  m <- intensity_obj$params$m
  funcs <- intensity_obj$intensity_funcs
  
  # Generate evaluation points
  t_values <- seq(0, t_max, length.out = resolution)
  
  # Calculate maximum intensity for y-axis
  max_intensity <- max(sapply(funcs, function(f) max(f(t_values))))
  
  # Setup plot canvas
  plot(NA, xlim = c(0, t_max), ylim = c(0, max_intensity),
       xlab = "Time", ylab = "Intensity", main = plot_title)
  
  # Plot each cluster's intensity
  for(i in 1:m) {
    lines(t_values, funcs[[i]](t_values), 
          col = colors[(i-1) %% length(colors) + 1], 
          lwd = 2)
  }
  
  # Add legend
  legend("topright", legend = paste("Cluster", 1:m), 
         col = colors[1:m], lwd = 2)
}

# Example usage -----------------------------------------------------------
if (FALSE) {  # Prevent execution when sourcing
  # Generate intensity functions
  intensity_obj <- generate_lambda0_intensity(
    K = 5,
    m = 3,
    total_time = 20,
    period_length = 10,
    seed = 42,
    periodic_weight = 0.9
  )
  
  # Visualize results
  plot_intensity_functions(intensity_obj)
}
