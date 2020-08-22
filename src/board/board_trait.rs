pub trait BoardInfo {
    fn step(&self) -> (String, Vec<usize>, f32);
    fn get_board_position(&self) -> [i8;36];
    fn print_board(&self); 
    fn get_possible_positions(&self) -> (Vec<String>, Vec<usize>);
    fn get_possible_moves(&self) -> Vec<usize>;
}
