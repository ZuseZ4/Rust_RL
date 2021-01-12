#[derive(Clone)]
pub struct Observation<T>
where
    T: std::clone::Clone,
{
    pub s0: T,
    pub a: usize,
    pub s1: T,
    pub r: f32,
    pub d: bool,
}

impl<T: std::clone::Clone> Observation<T> {
    pub fn new(s0: T, a: usize, s1: T, r: f32, d: bool) -> Self {
        Observation { s0, a, s1, r, d}
    }
}
