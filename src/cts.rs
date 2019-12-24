use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

fn log(f: f64) -> f64 {
    f.log(std::f64::consts::E)
}

fn log_add(logx: f64, logy: f64) -> f64 {
    let (logx, logy) = if logx > logy {
        (logy, logx)
    } else {
        (logx, logy)
    };
    let delta = logy - logx;
    if logx < 50.0 {
        log(delta.exp() + 1.0) + logx
    } else {
        logy
    }
}

const PRIOR_STAY_PROB: f64 = 0.5;
const PRIOR_SPLIT_PROB: f64 = 0.5;

#[derive(Error, Debug)]
pub enum CtsError {
    #[error("Unsupported prior: {0}")]
    UnsupportedPrior(String),
    #[error("Invalid context length: {0}")]
    InvalidContext(usize),
    #[error("Too many symbols")]
    TooManySymbols,
    #[error("Invalid Sampling")]
    InvalidSampling,
}

#[derive(Clone, Debug)]
enum EsitimatorPrior {
    Laplace,
    Jeffreys,
    Perks,
}

impl EsitimatorPrior {
    fn from_str(s: &str) -> Result<Self, CtsError> {
        match s {
            "laplace" => Ok(Self::Laplace),
            "jeffreys" => Ok(Self::Jeffreys),
            "perks" => Ok(Self::Perks),
            _ => Err(CtsError::UnsupportedPrior(s.to_string())),
        }
    }
    fn calc_prior(&self, symbol_size: u32) -> f64 {
        match self {
            Self::Laplace => 1.0,
            Self::Jeffreys => 0.5,
            Self::Perks => 1.0 / f64::from(symbol_size),
        }
    }
}

pub trait Symbol: Copy + Eq + std::hash::Hash {}

impl<T: Copy + Eq + std::hash::Hash> Symbol for T {}

#[derive(Debug)]
struct Esitimator<S: Symbol> {
    counts: HashMap<S, f64>,
    total_count: f64,
}

impl<S: Symbol> Esitimator<S> {
    fn new(count_init: f64) -> Self {
        Self {
            counts: HashMap::new(),
            total_count: count_init,
        }
    }
    fn _prob(&self, symbol: S, info: &UpdateInfo) -> f64 {
        let count = self
            .counts
            .get(&symbol)
            .map(|x| *x)
            .unwrap_or(info.symbol_prior);
        count / self.total_count
    }
    fn update(&mut self, symbol: S, info: &UpdateInfo) -> f64 {
        let counter = self.counts.entry(symbol).or_insert(info.symbol_prior);
        let res = *counter / self.total_count;
        *counter += 1.0;
        self.total_count += 1.0;
        log(res)
    }
    fn sample(&self, rng: &mut impl Rng) -> Option<S> {
        if self.counts.len() == 0 {
            return None;
        }
        loop {
            if let Some(symbol) = self.sample_once(rng) {
                return Some(symbol);
            }
        }
    }
    fn sample_once(&self, rng: &mut impl Rng) -> Option<S> {
        let mut random_index = rng.gen_range(0.0, f64::from(self.total_count));
        for (&symbol, &count) in self.counts.iter() {
            let count = f64::from(count);
            if random_index < count {
                return Some(symbol);
            }
            random_index -= count;
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct UpdateInfo {
    log_alpha: f64,
    log_1_min_alpha: f64,
    esitimator_init: f64,
    symbol_prior: f64,
}

impl UpdateInfo {
    fn new<S: Symbol>(cts: &Cts<S>) -> Self {
        Self {
            log_alpha: cts.log_alpha,
            log_1_min_alpha: cts.log_1_min_alpha,
            esitimator_init: f64::from(cts.symbol_size) * cts.symbol_prior,
            symbol_prior: cts.symbol_prior,
        }
    }
}

struct Node<S: Symbol> {
    children: HashMap<S, Self>,
    log_stay_prob: f64,
    log_split_prob: f64,
    estimator: Esitimator<S>,
}

impl<S: Symbol> Node<S> {
    fn new(esitimator_init: f64) -> Self {
        Node {
            children: HashMap::new(),
            log_stay_prob: log(PRIOR_STAY_PROB),
            log_split_prob: log(PRIOR_SPLIT_PROB),
            estimator: Esitimator::new(esitimator_init),
        }
    }
    fn update(&mut self, context: &[S], symbol: S, info: &UpdateInfo) -> f64 {
        let lp_estimator = self.estimator.update(symbol, info);
        if let Some((last, rest)) = context.split_last() {
            let child = self
                .children
                .entry(*last)
                .or_insert_with(|| Node::new(info.esitimator_init));
            let lp_child = child.update(&rest, symbol, info);
            let lp_node = self.mix_prediction(lp_estimator, lp_child);
            self.update_weights(lp_estimator, lp_child, info);
            lp_node
        } else {
            self.log_stay_prob = 0.0;
            lp_estimator
        }
    }
    fn mix_prediction(&self, lp_estimator: f64, lp_child: f64) -> f64 {
        let numerator = log_add(
            lp_estimator + self.log_stay_prob,
            lp_child + self.log_split_prob,
        );
        let denominator = log_add(self.log_stay_prob, self.log_split_prob);
        numerator - denominator
    }
    fn update_weights(&mut self, lp_estimator: f64, lp_child: f64, info: &UpdateInfo) {
        if info.log_1_min_alpha <= 0.0 {
            self.log_stay_prob += lp_estimator;
            self.log_split_prob += lp_child;
        } else {
            self.log_stay_prob = log_add(
                info.log_1_min_alpha + lp_estimator + self.log_stay_prob,
                info.log_alpha + lp_child + self.log_split_prob,
            );
            self.log_split_prob = log_add(
                info.log_1_min_alpha + lp_child + self.log_split_prob,
                info.log_alpha + lp_estimator + self.log_stay_prob,
            );
        }
    }
    fn _log_prob(&self, context: &[S], symbol: S, info: &UpdateInfo) -> f64 {
        let lp_estimator = log(self.estimator._prob(symbol, info));
        if let Some((_last, rest)) = context.split_last() {
            let lp_child = self
                .children
                .get(&symbol)
                .map(|c| c._log_prob(&rest, symbol, info))
                .unwrap_or(0.0);
            self.mix_prediction(lp_estimator, lp_child)
        } else {
            lp_estimator
        }
    }
    fn sample(&self, context: &[S], rng: &mut impl Rng) -> Option<S> {
        if let Some((last, rest)) = context.split_last() {
            let log_prob_stay =
                self.log_stay_prob - log_add(self.log_stay_prob, self.log_split_prob);
            if rng.gen_range(0.0, 1.0) < log_prob_stay.exp() {
                self.estimator.sample(rng)
            } else {
                let child = self.children.get(&last)?;
                child.sample(&rest, rng)
            }
        } else {
            self.estimator.sample(rng)
        }
    }
}

pub struct Cts<S: Symbol> {
    observed_data: HashSet<S>,
    context_length: usize,
    log_alpha: f64,
    log_1_min_alpha: f64,
    symbol_size: u32,
    symbol_prior: f64,
    time: f64,
    root: Node<S>,
}

impl<S: Symbol> Cts<S> {
    const MAX_REJECTTIONS: usize = 25;

    pub fn new(context_length: usize, max_symbol_size: u32, prior: &str) -> Result<Self, CtsError> {
        let prior = EsitimatorPrior::from_str(prior)?;
        let symbol_prior = prior.calc_prior(max_symbol_size);
        Ok(Cts {
            observed_data: HashSet::new(),
            context_length,
            log_alpha: 0.0,
            log_1_min_alpha: 0.0,
            symbol_size: max_symbol_size,
            symbol_prior,
            time: 0.0,
            root: Node::new(f64::from(max_symbol_size) * symbol_prior),
        })
    }

    fn check_context(&self, context: &[S]) -> Result<(), CtsError> {
        if context.len() != self.context_length {
            Err(CtsError::InvalidContext(context.len()))
        } else {
            Ok(())
        }
    }

    pub fn update(&mut self, context: &[S], symbol: S) -> Result<f64, CtsError> {
        self.time += 1.0;
        self.log_alpha = log(1.0 / (self.time + 1.0));
        self.log_1_min_alpha = log(self.time / (self.time + 1.0));
        self.check_context(context)?;
        self.observed_data.insert(symbol);
        if self.observed_data.len() > self.symbol_size as usize {
            return Err(CtsError::TooManySymbols);
        }
        Ok(self.root.update(context, symbol, &UpdateInfo::new(self)))
    }
    fn _log_prob(&self, context: &[S], symbol: S) -> Result<f64, CtsError> {
        self.check_context(context)?;
        Ok(self.root._log_prob(context, symbol, &UpdateInfo::new(self)))
    }
    pub fn sample(&self, context: &[S]) -> Result<S, CtsError> {
        if self.time == 0.0 {
            return Err(CtsError::InvalidSampling);
        }
        self.check_context(context)?;
        for _ in 0..Self::MAX_REJECTTIONS {
            if let Some(sym) = self.root.sample(context, &mut thread_rng()) {
                return Ok(sym);
            }
        }
        Ok(self.root.estimator.sample(&mut thread_rng()).unwrap())
    }
}

#[test]
fn test_cts() {
    let mut cts = Cts::<u8>::new(5, 26, "perks").unwrap();
    let log_p = cts.update(&&b"abcde"[..], b'f').unwrap();
    assert!(!log_p.is_nan());
    assert_eq!(cts.sample(&&b"abcde"[..]).unwrap(), b'f');
}
