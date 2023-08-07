use crate::utilities::calculate_rmse;
use crate::utilities::squared_euclidean_distance;

use super::interpolator::{Interpolator, build_design_matrix};
use super::linear_algebra::naive_linear_solver;
use super::kernels::gaussian_kernel;
use super::utilities::merge_unique_points;
// use super::linear_algebra::wrapped_ndarray_solver;

/// Simple Rbf model.
#[derive(Debug, Clone)]
pub struct Rbf {
    /// Input points.
    x: Vec<Vec<f64>>,
    /// Output points.
    y: Vec<f64>,
    /// Kernel function.
    kernel: fn(f64, f64) -> f64,
    /// Kernel bandwidth parameter.
    pub epsilon: f64,
    /// Interpolation coefficients.
    weights: Vec<f64>,
}

impl Rbf {
    /// Instantiates a new `Rbf` instance.
    ///
    /// # Arguments
    /// * `x`: A n*m matrix containing the training data points.
    /// * `y`: A n vector containing the corresponding training output values.
    /// * `kernel`: An optional function that computes the kernel function value.
    ///             Will default to Gaussian kernel if `None` given.
    /// * `epsilon`: An optional bandwidth parameter for the kernel.
    ///              Defaults to 1. if `None` given.
    ///
    /// # Returns
    /// A new `Rbf` instance.
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        kernel: Option<fn(f64, f64) -> f64>,
        epsilon: Option<f64>,
    ) -> Self {
        // check if a kernel is provided otherwise default to gaussian_kernel
        let kernel = kernel.unwrap_or(gaussian_kernel);

        // default bandwith/smoothing to 1. if not provided
        let epsilon = epsilon.unwrap_or(1.);

        // Evaluating linear system
        let design_matrix = build_design_matrix(&x, &x, &kernel, epsilon);

        // Solving linear system (can parallelise the naive solver)
        // let weights = match wrapped_ndarray_solver(&design_matrix, &y) {
        let weights = match naive_linear_solver(&design_matrix, &y) {
            Ok(weights) => weights,
            Err(msg) => panic!("{}", msg),
        };

        Rbf {
            x,
            y,
            kernel,
            epsilon,
            weights,
        }
    }

    /// Calibrates the model hyperparameters using a validation dataset.
    /// See Interpolator.
    ///
    /// # Note
    /// Tunes the hyperparameters [Naive until optimisation added]
    /// When calibrating you generally do not need to adjust the
    /// coefficients only the hyperparameters
    pub fn calibrate(&mut self, x_valid: &[Vec<f64>], y_valid: &[f64]) {
        // Calculate RMSE with current epsilon using both training and validation data
        let y_pred_train = self.predict(&self.x).unwrap();
        let y_pred_valid = self.predict(x_valid).unwrap();
        let mut best_rmse = (calculate_rmse(&y_pred_train, &self.y)
            + calculate_rmse(&y_pred_valid, y_valid))
            / 2.0;
        let mut best_epsilon = self.epsilon;

        // Optimizing with binary search
        let mut lo = 0.1;
        let mut hi = 1.0;
        let mut mid = (lo + hi) / 2.0;
        let diff = 0.001;
        let mut rmse: f64;

        while lo < hi {
            self.epsilon = mid;
            let y_pred_train = self.predict(&self.x).unwrap();
            let y_pred_valid = self.predict(x_valid).unwrap();
            rmse = (calculate_rmse(&y_pred_train, &self.y)
                + calculate_rmse(&y_pred_valid, y_valid))
                / 2.0;
            if rmse < best_rmse {
                best_rmse = rmse;
                best_epsilon = mid;
            }
            if rmse > best_rmse {
                hi = mid - diff;
            } else {
                lo = mid + diff;
            }
            mid = (lo + hi) / 2.0;
        }

        self.epsilon = best_epsilon;
    }
}

impl Interpolator for Rbf {
    /// Predicts the output values for a set of input data points using the RBF model.
    /// See Interpolator.
    fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let n = self.x.len();
        let mut result = vec![0.0; x_new.len()];
        for (k, x) in x_new.iter().enumerate() {
            for i in 0..n {
                let dist =
                    squared_euclidean_distance(x, &self.x[i]).max(std::f64::EPSILON);
                result[k] += self.weights[i] * (self.kernel)(dist, self.epsilon);
            }
            if result[k].is_nan() {
                return Err(String::from("NaN values in output"));
            }
        }
        Ok(result)
    }

    /// Updates the RBF model with new input and output data points.
    /// See Interpolator.
    fn update(&mut self, x_new: &[Vec<f64>], y_new: &[f64]) {
        merge_unique_points(&mut self.x, &mut self.y, x_new, y_new);

        let design_matrix =
            build_design_matrix(&self.x, &self.x, &self.kernel, self.epsilon);

        // self.weights = match wrapped_ndarray_solver(&design_matrix, &self.y) {
        self.weights = match naive_linear_solver(&design_matrix, &self.y) {
            Ok(weights) => weights,
            Err(msg) => panic!("{}", msg),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let rbf = Rbf::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![5.0, 6.0],
            None,
            None,
        );

        let x_new = vec![vec![2.5, 3.5]];
        let prediction = rbf.predict(&x_new).unwrap();

        // Compare the predicted result with the expected result
        assert_eq!(prediction, vec![5.11861403058931]);
    }
}
