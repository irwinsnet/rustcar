pub mod policy {

    pub struct Policy {
        pub max1: u8,
        pub max2: u8,
        pub max_move: u8,
        pub action_value: ndarray::Array3<f64>,
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
}