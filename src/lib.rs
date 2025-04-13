#![allow(unused)]
use std::cmp;

pub mod cars;
pub mod policy;
pub mod solver;


pub fn learn(mut agency: cars::RentalAgency) {
    let mut pi = policy::Policy::build_from_agency(&agency);
    
    // Estimate values for all states and actions.
    for s1 in solver::StateIterator::new(agency.max1, agency.max2) {
        let min_move = -1 * cmp::min(
            cmp::min(agency.max_move, s1.n2),
            agency.max1 - s1.n1
        ) as i8;
        let max_move = cmp::min(
            cmp::min(agency.max_move, s1.n1),
            agency.max2 - s1.n2) as i8;
        for a in min_move..(max_move + 1) {
            let val = agency.calc_value_for_action(&s1, a, &pi);
            pi.set_value(s1.n1, s1.n2, a, val);
            println!("State: {s1}, Action: {a}, Value: {val}")
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_actions() {
        // Arrange
        let cprobs = cars::RentalAgency::new(
            3, 1.0, 1.0, 3, 1.0, 1.0, 1);
        learn(cprobs);
    }
}