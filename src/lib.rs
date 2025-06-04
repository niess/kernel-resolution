use pyo3::prelude::*;

pub mod namespace;
pub mod numpy;
pub mod processor;


/// Gaussian kernel utilities.
#[pymodule]
fn kernel_resolution(module: &Bound<PyModule>) -> PyResult<()> {
    let py = module.py();

    // Initialise the NumPy interface.
    numpy::initialise(py)?;

    // Register class object(s).
    module.add_class::<processor::Processor>()?;

    Ok(())
}
