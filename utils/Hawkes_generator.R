library(splines2)
library(dplyr)
library(ggplot2)

source("utils/period_basis_intensity.R")

generate_hawkes_parameters <- function(
    m = 5,                      # Number of event types
    K = 5,                      # B-spline basis count
    period_length = 10,         # Period length for intensity functions
    theta_range = c(0, 0.5),    # Range for theta parameters in lambda0
    gamma_range = c(0, 1),      # Range for gamma parameters in lambda0
    alpha_range = c(0, 0.5),    # Range for alpha parameters
    beta_range = c(1, 2),       # Range for beta parameters
    seed = 42) {
  # Generate parameters for mutual Hawkes process

  
  if(!is.null(seed)) set.seed(seed)
  
  list(
    m = m,
    K = K,
    period_length = period_length,
    lambda0_theta = runif(K, theta_range[1], theta_range[2]),
    lambda0_gamma = matrix(runif(m*K, gamma_range[1], gamma_range[2]), nrow = m),
    alpha = matrix(runif(m*m, alpha_range[1], alpha_range[2]), nrow = m),
    beta = runif(m, beta_range[1], beta_range[2])
  )
}

normalize_alpha_matrix <- function(alpha, beta) {
  # Ensure spectral radius of alpha/beta matrix < 1
  alpha_beta <- alpha / beta
  svd_decomp <- svd(alpha_beta)
  max_singular <- max(svd_decomp$d)
  
  if (max_singular > 1) {
    scaling_factor <- max_singular * 1.5  # Conservative scaling
    alpha <- alpha / scaling_factor
  }
  return(alpha)
}

simulate_mutual_hawkes <- function(
    params,                  # Parameters from generate_hawkes_parameters
    total_time,              # Total simulation time
    memory_cutoff = 5,       # Memory window A
    seed = NULL) {
  # Simulate mutual Hawkes process with periodic intensities

  
  if(!is.null(seed)) set.seed(seed)
  
  # Generate intensity functions using previous implementation
  intensity_obj <- generate_lambda0_intensity(
    K = params$K,
    m = params$m,
    total_time = total_time,
    period_length = params$period_length,
    lambda0_gamma_range = c(min(params$lambda0_gamma), max(params$lambda0_gamma)),
    lambda0_theta_range = c(min(params$lambda0_theta), max(params$lambda0_theta)),
    seed = seed
  )
  
  # Initialize event storage
  event_df <- data.frame(event_time = numeric(), event_type = integer())
  
  # Generate immigrant events using thinned algorithm
  for(cluster in 1:params$m) {
    cluster_func <- intensity_obj$intensity_funcs[[cluster]]
    max_intensity <- find_max_lambda0(cluster_func, params$period_length)
    
    if(max_intensity <= 0) next
    
    # Optimized thinning implementation
    n_candidates <- max(1, rpois(1, 1.1 * max_intensity * total_time))
    candidates <- data.frame(
      time = runif(n_candidates, 0, total_time),
      prob = runif(n_candidates)
    )
    
    accepted <- candidates %>%
      mutate(intensity = sapply(time, cluster_func)) %>%
      filter(prob < (intensity / max_intensity))
    
    event_df <- bind_rows(event_df,
                         data.frame(event_time = accepted$time,
                                   event_type = cluster))
  }
  
  # Generate offspring events
  for(parent_cluster in 1:params$m) {
    parent_times <- event_df %>%
      filter(event_type == parent_cluster) %>%
      pull(event_time)
    
    if(length(parent_times) == 0) next
    
    for(child_cluster in 1:params$m) {
      effect <- params$alpha[parent_cluster, child_cluster] / 
               params$beta[parent_cluster]
      
      if(effect <= 0) next
      
      # Vectorized offspring generation
      offspring_counts <- rpois(length(parent_times), effect)
      valid_parents <- parent_times[offspring_counts > 0]
      valid_counts <- offspring_counts[offspring_counts > 0]
      
      offspring_times <- unlist(mapply(function(t, n) {
        t + rexp(n, params$beta[parent_cluster])
      }, valid_parents, valid_counts))
      
      # Apply memory cutoff
      valid_events <- offspring_times[offspring_times <= total_time &
                                      (offspring_times - rep(valid_parents, valid_counts)) <= memory_cutoff]
      
      if(length(valid_events) > 0) {
        event_df <- bind_rows(event_df,
                             data.frame(event_time = valid_events,
                                       event_type = child_cluster))
      }
    }
  }
  
  # Final processing
  event_df %>%
    filter(event_time <= total_time) %>%
    arrange(event_time) %>%
    distinct() %>%
    mutate(event_type = factor(event_type))
}

# Helper functions --------------------------------------------------------
find_max_lambda0 <- function(lambda_func, period, resolution = 1000) {
  # Find maximum baseline intensity
  
  grid <- seq(0, period, length.out = resolution)
  max(sapply(grid, lambda_func))
}

# Example usage 1-----------------------------------------------------------
if (FALSE) {
  # Generate parameters
  params <- generate_hawkes_parameters(
    m = 5,                      # Number of event types
    K = 5,                      # B-spline basis count
    period_length = 10,         # Period length for intensity functions
    theta_range = c(0, 0.5),    # Range for theta parameters in lambda0
    gamma_range = c(0, 1),      # Range for gamma parameters in lambda0
    alpha_range = c(0, 0.3),    # Range for alpha parameters
    beta_range = c(1, 5),       # Range for beta parameters
    seed = 42)
  
  # Normalize interaction matrix
  params$alpha <- normalize_alpha_matrix(params$alpha, params$beta)

  n <- 5000 # Training periods
  k <- 100    # Context window size
  n_test <- 100 # Test periods
  total_time <- (n + k + n_test) * params$period_length

  # Run simulation
  simulation <- simulate_mutual_hawkes(
    params = params,
    total_time = total_time,
    memory_cutoff = 5,
    seed = 45
  )

  print(head(simulation, n = 10))
  print(tail(simulation, n = 10))
  # write.csv(simulation, file = "simulation.csv", row.names = FALSE)
}


# Example usage 2-----------------------------------------------------------
if (TRUE) {
  output_dir <- "results2"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }

  # Generate parameters
  params <- generate_hawkes_parameters(
    m = 5,
    K = 5,
    period_length = 10,
    theta_range = c(0, 0.5),
    gamma_range = c(0, 1),
    alpha_range = c(0, 0.5),
    beta_range = c(1, 5),
    seed = 42
  )

  # Normalize interaction matrix
  params$alpha <- normalize_alpha_matrix(params$alpha, params$beta)

  n <- 5000 # Training periods
  k <- 100    # Context window size
  n_test <- 100 # Test periods
  total_time <- (n + k + n_test) * params$period_length

  # Run simulation for 100 times
  for (i in 1:100) {
    simulation <- simulate_mutual_hawkes(
      params = params,
      total_time = total_time,
      memory_cutoff = 5,
      seed = i
    )

    file_name <- sprintf("simulation_%03d.csv", i)
    file_path <- file.path(output_dir, file_name)

    write.csv(simulation, file = file_path, row.names = FALSE)
    
    cat(sprintf("Simulation %d saved to %s\n", i, file_path))
  }
}