use ndarray::Array1;
use rand::Rng;
pub fn get_random_true_entry(actions: Array1<bool>) -> usize {
    let num_legal_actions = actions.fold(0, |sum, &val| if val { sum + 1 } else { sum });
    assert!(num_legal_actions > 0, "no legal action available!");
    let mut action_number = rand::thread_rng().gen_range(0..num_legal_actions) as usize;
    let b = action_number;

    let mut position = 0;
    while (action_number > 0) | !actions[position] {
        if actions[position] {
            action_number -= 1;
        }
        position += 1;
    }
    assert!(
        actions[position],
        "randomly picked illegal move! {:} {} {}",
        actions, position, b
    );
    position
}
