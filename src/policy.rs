//! Policy Crate
//! 
//! In reinforcment learning, the policy is the set of actions that are chosen
//! for each state. The policy is represented by the `Policy` struct in the
//! `policy` module.

#![allow(unused)]

use std::{cmp, i8};
use crate::cars::RentalAgency;


/// Mapping of states to action.
/// 
/// The `action_value` field contains our current estimate of the value
/// of each state-action combination. The value is the expected value of
/// the sum of all subsequent rewards, assuming we follow the policy.
/// 
/// The `policy` field is a mapping of states to actions. The indices are
/// the number of cars at location 1 and location 2, and the array value
/// is an integer representing the number of cars to move from loc #1 to
/// loc #2. Negative actions indicate cars are moved from loc #2 to loc #1.
pub struct Policy {
    /// Maximum number of cars that can be kept at location #1
    pub max1: u8,
    /// Maximum number of cars that can be kept at location #2
    pub max2: u8,
    /// Maximum nmber of cars that can be moved between locations
    pub max_move: u8,
    /// Indexes are n1, n2
    pub value: ndarray::Array2<f64>,
    /// Indexes are n1, n2
    pub value_diff: f64,
    /// Maximum value difference between iterations k and k+1
    pub policy: ndarray::Array2<i8>,
    /// True if policy has changed
    pub policy_stable: bool
}

impl Policy {
    pub fn new(
        max1: u8, max2: u8, max_move: u8
    ) -> Policy {
        let total_moves = max_move * 2 + 1;
        let dimensions =
            ((max1 + 1) as usize, (max2 + 1) as usize);
        let value = 
            ndarray::Array2::<f64>::zeros(dimensions);
        let value_diff = f64::MAX;
        let policy =
            ndarray::Array2::<i8>::zeros(dimensions);
        let policy_stable = false;
            ndarray::Array2::<i32>::from_elem(dimensions, total_moves as i32);
        Policy {
            max1, max2, max_move, value, value_diff,
            policy, policy_stable
        }
    }

    pub fn build_from_agency(agency: &RentalAgency) -> Policy {
        Policy::new(agency.max1, agency.max2, agency.max_move)
    }

    pub fn get_value(&self, n1: u8, n2: u8) -> f64 {
        self.value[[n1 as usize, n2 as usize]]
    }

    pub fn set_value(&mut self, n1: u8, n2: u8, v: f64) {
        self.value[[n1 as usize, n2 as usize]] = v;
    }

    pub fn get_policy(&self, n1: u8, n2: u8) -> i8 {
        self.policy[[n1 as usize, n2 as usize]]
    }

    pub fn set_policy(&mut self, n1: u8, n2: u8, a: i8) {
        self.policy[[n1 as usize, n2 as usize]] = a;
    }


    /// Get move with highest value.
    /// 
    /// If all moves have zero value, best move is a = 0.
    pub fn get_best_move(&self, n1: u8, n2: u8) -> i8 {
        let min_move =
            -(cmp::min(
                cmp::min(n2, self.max_move) as i8, 
                (self.max1 - n1) as i8)
            );
        let max_move = 
        cmp::min(
            cmp::min(n1, self.max_move) as i8,
            (self.max2 - n2) as i8
        );
        let mut max_value = 0.0;
        let mut best_move: i8 = 0;
        for a in min_move..max_move + 1 {
            let value = self.get_value(
                ((n1 as i8) - a) as u8, ((n2 as i8) + a) as u8);
            if value > max_value {
                max_value = value;
                best_move = a;
            }
        }
        best_move
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_default_policy() {
        // Act
        let dpolicy = Policy::new(4, 4, 2);
        // Assert
        let vdims = dpolicy.value.dim();
        let pdims = dpolicy.policy.dim();
        assert_eq!(dpolicy.value.ndim(), 2);
        assert_eq!(vdims.0, 5);
        assert_eq!(vdims.1, 5);
        assert_eq!(dpolicy.value[[0, 0]], 0.0);
        assert_eq!(dpolicy.policy.ndim(), 2);
        assert_eq!(pdims.0, 5);
        assert_eq!(pdims.1, 5);
        assert_eq!(dpolicy.policy[[0, 0]], 0);
    }

    #[test]
    fn test_action_values() {
        // Arrange
        let dpolicy = Policy::new(4, 4, 2);
        // Act
        let best_move = dpolicy.get_best_move(1, 2);
        // Assert
        println!("Action: {}, Value: {}", best_move.0, best_move.1);
        assert!(best_move.0 >= -2 && best_move.0 <= 2);
        assert!(best_move.1 >= 0.0);
    }
}
