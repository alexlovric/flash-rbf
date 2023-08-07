use pyo3::prelude::*;

use flash_rbf_core::{
    rbf::Rbf,
    kernels::{
        gaussian_kernel, multiquadric_kernel, inverse_multi_kernel, linear_kernel,
        cubic_kernel,
    },
    interpolator::Interpolator,
};

/// Simple Rbf model.
#[derive(Clone, Debug)]
#[pyclass(name = "Rbf")]
pub struct PyRbf {
    pub wrapped_model: Rbf,
}

#[pymethods]
impl PyRbf {
    /// Instantiates a new `PyRbf` instance.
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
    #[new]
    pub fn new(
        x: &PyAny,
        y: &PyAny,
        kernel_name: Option<&str>,
        epsilon: Option<f64>,
    ) -> Self {
        let kernel_name = kernel_name.unwrap_or("gaussian");
        let kernel = match kernel_name {
            "gaussian" => gaussian_kernel,
            "multiquadric" => multiquadric_kernel,
            "inverse_multiquadratic" => inverse_multi_kernel,
            "linear" => linear_kernel,
            "cubic" => cubic_kernel,
            _ => panic!("Kernel not implemented"),
        };

        PyRbf {
            wrapped_model: Rbf::new(
                x.extract().unwrap(),
                y.extract().unwrap(),
                Some(kernel),
                epsilon,
            ),
        }
    }

    pub fn predict(&self, x_new: Vec<Vec<f64>>) -> Vec<f64> {
        Interpolator::predict(self, &x_new).unwrap()
    }

    pub fn update(&mut self, x_new: Vec<Vec<f64>>, y_new: Vec<f64>) {
        Interpolator::update(self, &x_new, &y_new)
    }

    pub fn calibrate(&mut self, x_valid: Vec<Vec<f64>>, y_valid: Vec<f64>) {
        self.wrapped_model.calibrate(&x_valid, &y_valid)
    }

    pub fn __repr__(&self) -> String {
        let mut ss = format!(
            "┌{}┐\n│{: <48}│\n╞{}╡\n│{: <48}│\n|{: <48}│\n",
            "─".repeat(48),
            "Rbf Model:",
            "═".repeat(48),
            "Kernel: Gaussian",
            format!("Epsilon: {}", self.wrapped_model.epsilon)
        );
        ss += &format!("└{}┘\n", "─".repeat(48));
        ss
    }
}

impl Interpolator for PyRbf {
    fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        self.wrapped_model.predict(x_new)
    }

    fn update(&mut self, x_new: &[Vec<f64>], y_new: &[f64]) {
        self.wrapped_model.update(x_new, y_new)
    }
}
