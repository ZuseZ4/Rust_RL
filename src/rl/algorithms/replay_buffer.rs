use crate::rl::algorithms::observation::Observation;
use rand::Rng;

/// A struct containing all relevant information.
pub struct ReplayBuffer<T>
where
    T: std::clone::Clone,
{
    b: usize,                           // Batch_size
    m_max: usize,                       // #Memories to store
    iter_pos: usize,                    // to be able to remove the oldest entries
    is_full: bool,                      // how full is the storage
    memories: Vec<Box<Observation<T>>>, // up to M observations
}

impl<T: std::clone::Clone> ReplayBuffer<T> {
    /// Creates a new replay buffer.
    ///
    /// B is the amount of observations to be retreived by calling get_memories().
    /// M is the maximum amount of observations to be stored simultaneously.
    pub fn new(b: usize, m: usize) -> Self {
        assert!(m>0,"a replay buffer without the possibility to store memories doesn't make sense. Please increase m!");
        ReplayBuffer {
            b,
            m_max: m,
            iter_pos: 0,
            is_full: false,
            memories: Vec::with_capacity(b),
        }
    }

    /// Returns true if the replay buffer is filled entirely.
    ///
    /// New entries can still be added, however they will replace the oldest entry.
    pub fn is_full(&self) -> bool {
        self.is_full
    }

    /// Returns true if no entry has been stored yet.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Get a vector containing B Observations.
    ///
    /// Panics if no memory has previously been added.
    /// May return a single memory multiple times, mainly if only few observations have been added.
    pub fn get_memories(&self) -> Vec<Box<Observation<T>>> {
        assert!(
            self.is_full || self.iter_pos > 0,
            "add at least a single observation before calling get_memories()"
        );
        let mut rng = rand::thread_rng();
        let mut res: Vec<Box<Observation<T>>> = Vec::new();
        let max = if self.is_full {
            self.m_max
        } else {
            self.iter_pos
        };
        for _ in 0..self.b {
            let index = rng.gen_range(0, max);
            res.push(self.memories[index].clone());
        }
        res
    }

    /// Add a single memory to our replay buffer.
    ///
    /// If the maximum amount of entries is already reached, the oldest entry is replaced.
    /// Otherwise our new entry is appended.
    pub fn add_memory(&mut self, obs: Observation<T>) {
        let Observation { r, .. } = obs;
        if r < 0.0001 && rand::thread_rng().gen::<f32>() < 0.2 {
            return;
        }
        if !self.is_full {
            self.memories.push(Box::new(obs));
        } else {
            self.memories[self.iter_pos] = Box::new(obs);
        }
        self.iter_pos += 1;
        if self.iter_pos == self.m_max {
            self.iter_pos = 0;
            self.is_full = true;
        }
    }
}
