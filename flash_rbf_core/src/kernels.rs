/// kernels (All kernels accept squared Euclidean for the `distance`)
/// Guassian
pub fn gaussian_kernel(distance: f64, bandwidth: f64) -> f64 {
    (-0.5 * distance / bandwidth.powi(2)).exp()
}

/// Multiquadratic
pub fn multiquadric_kernel(distance: f64, bandwidth: f64) -> f64 {
    (distance + bandwidth.powi(2)).sqrt()
}

/// Inverse
pub fn inverse_multi_kernel(distance: f64, bandwidth: f64) -> f64 {
    1.0 / multiquadric_kernel(distance, bandwidth)
}

/// Linear
pub fn linear_kernel(distance: f64, _bandwidth: f64) -> f64 {
    distance
}

/// Cubic
pub fn cubic_kernel(distance: f64, _bandwidth: f64) -> f64 {
    distance.powi(2)
}
