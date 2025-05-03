#![allow(unused)]
use std::cmp;
use ndarray_stats::QuantileExt;

pub mod cars;
pub mod policy;
pub mod solver;


pub fn learn_via_policy_iteration(agency: cars::RentalAgency) -> policy::Policy {
    let mut pi = policy::Policy::build_from_agency(&agency);
    let mut delta = f64::MAX;
    while delta > 0.01 {
        (pi, delta) = evaluate_policy(&agency, pi);
        println!("Value Delta: {}", delta);
    }
    let (mut improved_pi, stable) = improve_policy(&agency, pi);
    improved_pi
}

fn evaluate_policy(
    agency: &cars::RentalAgency,
    mut pi: policy::Policy
) -> (policy::Policy, f64) {
    let mut delta = f64::MAX;
    for s1 in solver::StateIterator::new(agency.max1, agency.max2) {
        let prior_value = pi.get_value(s1.n1, s1.n2);
        let new_value = agency.calc_value(&s1, &pi, None);
        pi.set_value(s1.n1, s1.n2, new_value);
        delta = f64::max(delta, (prior_value - new_value).abs());
    }
    (pi, delta)
}

fn improve_policy(
    agency: &cars::RentalAgency,
    mut pi: policy::Policy
) -> (policy::Policy, bool) {
    let mut policy_stable = true;
    let mut a: i8 = 0;
    for s1 in solver::StateIterator::new(agency.max1, agency.max2) {
        let old_a = pi.get_policy(s1.n1, s1.n2);
        a = pi.get_best_move(s1.n1, s1.n2);
        if a != old_a {
            pi.set_policy(s1.n1, s1.n2, a);
            policy_stable = false;
        }
    }
    (pi, policy_stable)
}



// pub fn learn(mut agency: cars::RentalAgency) -> policy::Policy {
//     let mut pi = policy::Policy::build_from_agency(&agency);
//     let mut value_diff = f64::MAX;
//     let mut policy_diff = i32::MAX;
//     while value_diff > 0.1  || policy_diff > 0 {
//         pi = update_values(&agency, pi);
//         value_diff = *pi.value_diff.max().unwrap();
//         policy_diff = pi.policy_diff.mapv(i32::abs).sum();
//         println!("Max value diff: {} | Total policy diff: {}", value_diff, policy_diff);
//         cars::RentalAgency::show_array(&pi.policy_diff, String::from("L2"));
//     }
//     pi
// }


// pub fn update_values(
//     agency: &cars::RentalAgency,
//     mut pi: policy::Policy
// ) -> policy::Policy {   
//     // Estimate values for all states and actions.
//     for s1 in solver::StateIterator::new(agency.max1, agency.max2) {
//         let prior_val = pi.get_value(s1.n1, s1.n2);
//         let val = agency.calc_value(&s1, &pi);
//         let curr_move = pi.get_policy(s1.n1, s1.n2);
//         let (best_move, max_value) = pi.get_best_move(s1.n1, s1.n2);
//         if max_value > val {
//             pi.set_policy(s1.n1, s1.n2, best_move);
//             pi.set_policy_diff(s1.n1, s1.n2, (best_move - curr_move) as i32);
//             pi.set_value(s1.n1, s1.n2, max_value);
//             pi.set_value_diff(s1.n1, s1.n2, max_value - prior_val);
//         } else {
//             pi.set_policy_diff(s1.n1, s1.n2, (best_move - curr_move) as i32);
//             pi.set_value(s1.n1, s1.n2,  val);
//             pi.set_value_diff(s1.n1, s1.n2, val - prior_val);
//         }
//     }
//     pi
// }


#[cfg(test)]
mod tests {
    use crate::cars::RentalAgency;

    use super::*;

    #[test]
    fn learn_actions() {
        // Arrange
        let agency = cars::RentalAgency::new(
            3, 1.0, 1.0, 3, 1.0, 1.0, 1);
        let pi = learn(agency);
    }
}