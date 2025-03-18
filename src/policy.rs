//! Policy Crate
//! 
//! In reinforcment learning, the policy is the set of actions that are chosen
//! for each state. The policy is represented by the `Policy` struct in the
//! `policy` module.

#![allow(unused)]


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
    /// Indexes are n1, n2, a + max_move
    pub action_value: ndarray::Array3<f64>,
    /// Indexes are n1, n2
    pub policy: ndarray::Array2<i8>
}

impl Policy {
    pub fn new(
        max1: u8, max2: u8, max_move: u8
    ) -> Policy {
        let total_moves = max_move * 2 + 1;
        let dimensions =
            ((max1 + 1) as usize, (max2 + 1) as usize, total_moves as usize);
        let action_value = 
            ndarray::Array3::<f64>::zeros(dimensions);
        let policy_array =
            ndarray::Array2::<i8>::zeros(
                ((max1 + 1) as usize, (max2 + 1) as usize));
        let policy = Policy {
            max1, max2, max_move, action_value, policy: policy_array
        };
        policy
    }

    pub fn get_value(&self, n1: u8, n2: u8, a: i8) -> f64 {
        let a_idx = (a + self.max_move as i8) as usize;
        self.action_value[[n1 as usize, n2 as usize, a_idx]]
    }

    pub fn set_value(&mut self, n1: u8, n2: u8, a: i8, v: f64) {
        let a_idx = (a + self.max_move as i8) as usize;
        self.action_value[[n1 as usize, n2 as usize, a_idx]] = v;
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
        let vdims = dpolicy.action_value.dim();
        let pdims = dpolicy.policy.dim();
        assert_eq!(dpolicy.action_value.ndim(), 3);
        assert_eq!(vdims.0, 5);
        assert_eq!(vdims.1, 5);
        assert_eq!(vdims.2, 5);
        assert_eq!(dpolicy.action_value[[0, 0, 0]], 0.0);
        assert_eq!(dpolicy.policy.ndim(), 2);
        assert_eq!(pdims.0, 5);
        assert_eq!(pdims.1, 5);
        assert_eq!(dpolicy.policy[[0, 0]], 0);
    }

}
