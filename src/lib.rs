#![allow(unused)]
use std::cmp;
use ndarray_stats::QuantileExt;

pub mod cars;
pub mod policy;
pub mod solver;


pub fn learn(mut agency: cars::RentalAgency) -> policy::Policy {
    let mut pi = policy::Policy::build_from_agency(&agency);
    let mut value_diff = f64::MAX;
    while value_diff > 0.1 {
        pi = update_values(&agency, pi);
        value_diff = *pi.value_diff.max().unwrap();
        println!("Max value diff: {}", value_diff);
    }
    pi
}


pub fn update_values(
    agency: &cars::RentalAgency,
    mut pi: policy::Policy
) -> policy::Policy {   
    // Estimate values for all states and actions.
    for s1 in solver::StateIterator::new(agency.max1, agency.max2) {
        let prior_val = pi.get_value(s1.n1, s1.n2);
        let val = agency.calc_value(&s1, &pi);
        let curr_move = pi.policy[[s1.n1 as usize, s1.n2 as usize]];
        let (best_move, max_value) = pi.get_best_move(s1.n1, s1.n2);
        if max_value > val {
            pi.policy[[s1.n1 as usize, s1.n2 as usize]] = best_move;
            pi.set_value(s1.n1, s1.n2, max_value);
            pi.set_value_diff(s1.n1, s1.n2, max_value - prior_val);
        } else {
            pi.set_value(s1.n1, s1.n2,  val);
            pi.set_value_diff(s1.n1, s1.n2, val - prior_val);
        }
    }
    pi
}


#[cfg(test)]
mod tests {
    use crate::cars::RentalAgency;

    use super::*;

    #[test]
    fn learn_actions() {
        // Arrange
        let agency = cars::RentalAgency::new(
            3, 1.0, 1.0, 3, 1.0, 1.0, 1);
    }
}