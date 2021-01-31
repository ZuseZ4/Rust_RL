#[derive(Default)]
pub struct RunningResults {
    n: usize,
    print: bool,
    full: bool,
    current_pos: usize,
    won: u32,
    draw: u32,
    lost: u32,
    results: Vec<i32>,
}

impl RunningResults {
    pub fn new(n: usize, print: bool) -> RunningResults {
        RunningResults {
            n,
            print,
            results: vec![0; n],
            ..Default::default()
        }
    }

    pub fn add(&mut self, result: i32) {
        if self.full {
            match self.results[self.current_pos] {
                -1 => self.lost -= 1,
                0 => self.draw -= 1,
                1 => self.won -= 1,
                _ => panic!(),
            }
        }
        self.results[self.current_pos] = result;
        self.current_pos = (self.current_pos + 1) % self.n;
        if self.current_pos == 0 {
            self.full = true;
        }
        match result {
            -1 => self.lost += 1,
            0 => self.draw += 1,
            1 => self.won += 1,
            _ => panic!(),
        }
        if self.current_pos == 0 && self.print {
            println!(
                "accumulated results of last {} epochs: \t lost: {} \t draw: {} \t won: {}",
                self.n, self.lost, self.draw, self.won
            );
        }
    }

    pub fn get_results(&self) -> (u32, u32, u32) {
        (self.lost, self.draw, self.won)
    }
}
