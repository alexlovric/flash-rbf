use pyo3::prelude::*;

pub mod py_rbf;
use py_rbf::PyRbf;

#[pymodule]
fn flash_rbf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRbf>()?;
    Ok(())
}
