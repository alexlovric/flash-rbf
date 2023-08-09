use crate::utilities::squared_euclidean_distance;

/// Can interpolate
pub trait Interpolator {
    /// Predict the output `y_new` given points `x_new`.
    ///
    /// # Arguments
    /// * `x_new`: New input data points to predict the output values for.
    ///
    /// # Returns
    /// A `Result` containing output values if the computation is successful.
    fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<f64>, String>;
    /// Update the interpolator with a given `x_new` and `y_new`.
    ///
    /// # Arguments
    /// * `x_new`: An array containing the new input data points to add to the model.
    /// * `y_new`: An array containing the corresponding output values.
    fn update(&mut self, x_new: &[Vec<f64>], y_new: &[f64]);
}

/// Builds a design matrix based on two sets of input vectors and a kernel function.
///
/// # Arguments
/// * `x0`: A slice of vectors representing the first set of input data.
/// * `x1`: A slice of vectors representing the second set of input data.
/// * `kernel`: A function that takes two f64 values and returns a f64 value.
/// * `epsilon`: The kernel function's parameter.
///
/// # Returns
/// A vector of vectors representing the design matrix.
pub fn build_design_matrix(
    x0: &[Vec<f64>],
    x1: &[Vec<f64>],
    kernel: &fn(f64, f64) -> f64,
    epsilon: f64,
) -> Vec<Vec<f64>> {
    (0..x0.len())
        .map(|i| {
            (0..x1.len())
                .map(|j| {
                    let dist =
                        squared_euclidean_distance(&x0[i], &x1[j]).max(std::f64::EPSILON);
                    kernel(dist, epsilon)
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
}
