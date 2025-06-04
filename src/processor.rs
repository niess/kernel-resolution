use crate::numpy::{AnyArray, ArrayMethods, NewArray, PyArray};
use crate::namespace::Namespace;
use pyo3::prelude::*;
use pyo3::exceptions::{PyStopIteration, PyTypeError};


#[pyclass(module="gaussian_kernel")]
pub struct Processor {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,

    #[pyo3(get)]
    pub xmin: f64,
    #[pyo3(get)]
    pub xmax: f64,
    #[pyo3(get)]
    pub nx: usize,

    pub events: usize,
    pub y1: Vec<f64>,
    pub y2: Vec<f64>,
}

#[pyclass(module="gaussian_kernel")]
pub struct Iter {
    collector: Py<Processor>,
    x: Py<PyArray<f64>>,
    w: PyWeights,
    events: usize,
    i: usize,
    m: usize,
}

#[derive(FromPyObject)]
enum AnyWeights<'py> {
    Scalar(f64),
    Array(AnyArray<'py, f64>),
}

enum BoundWeights<'py> {
    Scalar(f64),
    Array(Bound<'py, PyArray<f64>>),
}

enum PyWeights {
    Scalar(f64),
    Array(Py<PyArray<f64>>),
}

#[pymethods]
impl Processor {
    #[pyo3(signature=(/, *, coefficients, xmin, xmax, nx))]
    #[new]
    fn new(coefficients: Vec<f64>, xmin: f64, xmax: f64, nx: usize) -> Self {
        let y1 = vec![0.0; nx];
        let y2 = vec![0.0; nx];
        let events = 0;
        Self { coefficients, xmin, xmax, nx, y1, y2, events }
    }

    /// Export the processed density.
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut x_array = NewArray::<f64>::empty(py, [self.nx])?;
        let mut y_array = NewArray::<f64>::empty(py, [self.nx, 2])?;
        let x = x_array.as_slice_mut();
        let y = y_array.as_slice_mut();

        const PI: f64 = std::f64::consts::PI;

        let n = self.events as f64;
        let dx = (self.xmax - self.xmin) / ((self.nx - 1) as f64);
        for i in 0..self.nx {
            let xi = self.xmin + (i as f64) * dx;
            x[i] = xi;

            let s1 = self.y1[i] / ((2.0 * PI).sqrt() * n);
            let s2 = self.y2[i] / (2.0 * PI * n);

            y[2 * i] = s1;
            y[2 * i + 1] = ((s2 - s1.powi(2)).max(0.0) / n).sqrt();
        }

        Namespace::new(py, [
            ("x", x_array.into_bound()),
            ("y", y_array.into_bound()),
        ])
    }

    /// Iterator over the processing of bare x values.
    #[pyo3(signature=(x, /, *, weights=None, events=None))]
    fn iterator(
        slf: &Bound<Self>,
        x: AnyArray<f64>,
        weights: Option<AnyWeights>,
        events: Option<usize>,
    ) -> PyResult<Iter> {
        let events = events.unwrap_or_else(|| x.size());
        let weights = match weights {
            None => PyWeights::Scalar(1.0),
            Some(weights) => match weights {
                AnyWeights::Scalar(weights) => PyWeights::Scalar(weights),
                AnyWeights::Array(weights) => {
                    if x.size() != weights.size() {
                        return Err(PyTypeError::new_err(format!(
                            "incompatible x and weights sizes ({} != {})",
                            x.size(),
                            weights.size(),
                        )))
                    }
                    PyWeights::Array(weights.into_bound().unbind())
                }
            }
        };

        let n = x.size();
        let m = (n / 10000).max(1);
        let collector = slf.clone().unbind();
        let x = x.into_bound().unbind();

        Ok(Iter { collector, x, w: weights, events, i: 0, m })
    }

    /// Process bare x values.
    #[pyo3(signature=(x, /, *, weights=None, events=None))]
    fn process(
        &mut self,
        x: AnyArray<f64>,
        weights: Option<AnyWeights>,
        events: Option<usize>,
    ) -> PyResult<()> {
        let events = events.unwrap_or_else(|| x.size());
        let weights = match weights {
            None => BoundWeights::Scalar(1.0),
            Some(weights) => match weights {
                AnyWeights::Scalar(weights) => BoundWeights::Scalar(weights),
                AnyWeights::Array(weights) => {
                    if x.size() != weights.size() {
                        return Err(PyTypeError::new_err(format!(
                            "incompatible x and weights sizes ({} != {})",
                            x.size(),
                            weights.size(),
                        )))
                    }
                    BoundWeights::Array(weights.into_bound())
                }
            }
        };
        let end = x.size() - 1;
        self.collect_raw(&x.into_bound(), &weights, 0, end)?;
        self.events += events;

        Ok(())
    }

    /// Compute the resolution (sigma) at bare x values.
    #[pyo3(signature=(x, /))]
    fn sigma(&self, py: Python, x: AnyArray<f64>) -> PyResult<Py<PyArray<f64>>> {
        let mut sigma_array = NewArray::<f64>::empty(py, x.shape())?;
        let sigma = sigma_array.as_slice_mut();
        for i in 0..x.size() {
            let xi = x.get_item(i)?;
            sigma[i] = self.sigma1(xi);
        }

        Ok(sigma_array.into_bound().unbind())
    }
}

impl Processor {
    #[inline]
    fn sigma1(&self, x: f64) -> f64 {
        let mut c = self.coefficients.iter().rev();
        let mut sigma = c.next().copied().unwrap();
        let mut xi = 1.0;
        let sx = x.sqrt();
        for ci in c {
            xi *= sx;
            sigma += *ci * xi;
        }
        sigma
    }

    fn collect_raw<'py>(
        &mut self,
        x: &Bound<'py, PyArray<f64>>,
        w: &BoundWeights,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let dx = (self.xmax - self.xmin) / ((self.nx - 1) as f64);
        for i in start..=end {
            let wi = match w {
                BoundWeights::Scalar(w) => *w,
                BoundWeights::Array(w) => w.get_item(i)?,
            };
            if wi <= 0.0 {
                continue
            }
            let xi = x.get_item(i)?;
            let sigma = self.sigma1(xi);
            let j0 = (((xi - 5.0 * sigma - self.xmin) / dx).floor() as usize).max(0);
            let j1 = (((xi + 5.0 * sigma - self.xmin) / dx).floor() as usize).min(self.nx - 1);

            for j in j0..=j1 {
                let xj = self.xmin + (j as f64) * dx;
                let gij = (-0.5 * ((xi - xj) / sigma).powi(2)).exp() / sigma;
                let wij = gij * wi;
                self.y1[j] += wij;
                self.y2[j] += wij.powi(2);
            }
        }
        Ok(())
    }
}

#[pymethods]
impl Iter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __len__(&self, py: Python) -> usize {
        let x = self.x.bind(py);
        let n = x.size();
        if n % self.m == 0 {
            n / self.m
        } else {
            n / self.m + 1
        }
    }

    fn __next__(&mut self, py: Python) -> PyResult<usize> {
        let x = self.x.bind(py);
        let n = x.size().min(self.i + self.m);
        if self.i >= n {
            return Err(PyStopIteration::new_err(()))
        }

        let w = self.w.bind(py);
        let mut collector = self.collector.bind(py).borrow_mut();
        collector.collect_raw(x, &w, self.i, n - 1)?;

        let index = self.i / self.m;
        self.i = n;
        if n >= x.size() {
            collector.events += self.events;
        }

        Ok(index)
    }
}

impl PyWeights {
    fn bind<'py>(&self, py: Python<'py>) -> BoundWeights<'py> {
        match self {
            Self::Scalar(w) => BoundWeights::Scalar(*w),
            Self::Array(w) => BoundWeights::Array(w.bind(py).clone()),
        }
    }
}
