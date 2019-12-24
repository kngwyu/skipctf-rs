use pyo3::prelude::*;

mod cts;

fn convert_result<T>(res: Result<T, cts::CtsError>) -> PyResult<T> {
    res.map_err(|e| PyErr::new::<pyo3::exceptions::RuntimeError, _>(format!("{}", e)))
}

#[pyclass]
pub struct SequencePredictor {
    context: Vec<u8>,
    cts: cts::Cts<u8>,
}

#[pymethods]
impl SequencePredictor {
    #[new]
    fn new(context_length: usize, symbol_size: u32, initial_symbol: u8) -> PyResult<Self> {
        let mut context = Vec::new();
        for _ in 0..context_length {
            context.push(initial_symbol);
        }
        convert_result(
            cts::Cts::new(context_length, symbol_size, "perks")
                .map(|cts| SequencePredictor { context, cts: cts }),
        )
    }

    fn train(&mut self, text: &[u8]) -> PyResult<f64> {
        let mut log_prob = 0.0;
        for (i, &ch) in text.iter().enumerate() {
            let context = &self.context[i..];
            log_prob += convert_result(self.cts.update(context, ch))?;
            self.context.push(ch);
        }
        self.context = self.context.split_off(text.len());
        Ok(log_prob)
    }

    fn sample(&mut self, n_samples: usize) -> PyResult<String> {
        let ctx_len = self.context.len();
        for i in 0..n_samples {
            let sampled = convert_result(self.cts.sample(&self.context[i..]))?;
            self.context.push(sampled);
        }
        let result = self.context.split_off(ctx_len);
        String::from_utf8(result).map_err(Into::into)
    }
}

#[pymodule]
fn skipcts(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SequencePredictor>()
}
