pub trait Executor {
    fn run_boxed(&self, f: Box<dyn FnOnce() + Send + 'static>);
    fn run<F: FnOnce() + Send + 'static>(&self, f: F)
    where
        Self: Sized,
    {
        self.run_boxed(Box::new(f));
    }
}

#[derive(Default)]
pub struct BasicExecutor;

impl Executor for BasicExecutor {
    fn run_boxed(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        std::thread::spawn(f);
    }
}

#[cfg(feature = "rayon")]
pub struct RayonExecutor {
    pool: rayon::ThreadPool,
}

#[cfg(feature = "rayon")]
impl RayonExecutor {
    pub fn new(num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build thread pool");
        Self { pool }
    }
}

#[cfg(feature = "rayon")]
impl Executor for RayonExecutor {
    fn run_boxed(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        self.pool.spawn(f);
    }
}
