use rayon::{ThreadPool, ThreadPoolBuilder};

/// Context provided to jobs running on worker threads.
#[derive(Default)]
pub struct ThreadCtx;

/// A job to be executed by the dispatcher.
pub type Job = Box<dyn FnOnce(&mut ThreadCtx) + Send + 'static>;

/// Per-frame queue of jobs.
pub struct FrameCtx {
    jobs: Vec<Job>,
}

impl FrameCtx {
    fn new() -> Self {
        Self { jobs: Vec::new() }
    }
}

/// Statistics about dispatched jobs.
#[derive(Default)]
pub struct JobDispatchStats {
    /// Total number of jobs submitted.
    pub jobs_submitted: usize,
}

/// Dispatcher responsible for executing jobs across frames and threads.
pub struct JobDispatch {
    executor: ThreadPool,
    frames: Vec<FrameCtx>,
    curr_frame: usize,
    stats: JobDispatchStats,
}

impl JobDispatch {
    /// Create a new dispatcher with the given number of worker threads and frames in flight.
    pub fn new(num_threads: usize, num_frames: usize) -> Self {
        let executor = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build thread pool");
        let frames = (0..num_frames).map(|_| FrameCtx::new()).collect();
        Self {
            executor,
            frames,
            curr_frame: 0,
            stats: JobDispatchStats::default(),
        }
    }

    /// Begin work on the specified frame.
    pub fn begin_frame(&mut self, frame_idx: usize) {
        self.curr_frame = frame_idx % self.frames.len();
    }

    /// Enqueue a job for the current frame.
    pub fn enqueue<F>(&mut self, job: F)
    where
        F: FnOnce(&mut ThreadCtx) + Send + 'static,
    {
        self.frames[self.curr_frame].jobs.push(Box::new(job));
    }

    /// Submit all queued jobs for the current frame to the executor.
    pub fn submit(&mut self) {
        let jobs = std::mem::take(&mut self.frames[self.curr_frame].jobs);
        for job in jobs {
            self.stats.jobs_submitted += 1;
            self.executor.spawn(move || {
                let mut ctx = ThreadCtx::default();
                job(&mut ctx);
            });
        }
    }

    /// Recycle resources associated with the specified frame.
    pub fn recycle(&mut self, frame_idx: usize) {
        let idx = frame_idx % self.frames.len();
        self.frames[idx].jobs.clear();
    }

    /// Access statistics about dispatched jobs.
    pub fn stats(&self) -> &JobDispatchStats {
        &self.stats
    }
}

