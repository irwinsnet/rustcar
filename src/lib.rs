
mod policy;
mod solver;


pub mod cars {

    use statrs::distribution::{Discrete, DiscreteCDF, Poisson};
    
    // use statrs::statistics::Data;

    /// Car rental and return probabilities.
    /// 
    /// Precalculate rental and return probabilities when object is constructed.
    /// Indices to probability tables x1, y1, x2, and y2 are
    /// [cars on lot, number of cars rented or returned].
    pub struct CarProbs {
        pub max1: u8,
        pub rent_mean1: f32,
        pub return_mean1: f32,
        pub x1: ndarray::Array2<f64>,  // Site 1 rental probabilities
        pub y1: ndarray::Array2<f64>,  // Site 1 return probabilities
        pub max2: u8,
        pub rent_mean2: f32,
        pub return_mean2: f32,
        pub x2: ndarray::Array2<f64>,  // Site 2 rental probabilities
        pub y2: ndarray::Array2<f64>,  // Site 2 return probabilities
        pub max_move: u8,
    }

    impl CarProbs {
        pub fn new(
            max1: u8, rent_mean1: f32, return_mean1: f32,
            max2: u8, rent_mean2: f32, return_mean2: f32,
            max_move: u8,
        ) -> CarProbs {
            let x1_probs =
                CarProbs::calc_rent_probs(rent_mean1, max1);
            let y1_probs = 
                CarProbs::calc_return_probs(return_mean1, max1);
            let x2_probs =
                CarProbs::calc_rent_probs(rent_mean2, max2);
            let y2_probs = 
                CarProbs::calc_return_probs(return_mean2, max2);

            CarProbs {
                max1, rent_mean1, return_mean1,
                x1: x1_probs, y1: y1_probs,
                max2, rent_mean2, return_mean2,
                x2: x2_probs, y2: y2_probs,
                max_move
            }
        }

        fn calc_rent_probs(mean: f32, max_n: u8) -> ndarray::Array2<f64> {
            let rent_dist = Poisson::new(f64::from(mean)).unwrap();
            let dim = (max_n + 1) as usize;
            let mut x_probs =
                ndarray::Array2::<f64>::zeros((dim, dim));
            for n in 0..max_n + 1 {
                for x in 0..max_n + 1 {
                    x_probs[[n as usize, x as usize]] = 
                        CarProbs::rent_prob(n, x, max_n, &rent_dist);
                }
            }
            x_probs
        }

        fn rent_prob(n: u8, x: u8, max_n: u8, rent_dist: &Poisson) -> f64 {
            // Can't fit more than max_n cars on lot.
            if n > max_n {
                return 0.0;
            }
            // No cars on lot, so only zero rentals allowed.
            if n == 0 && x == 0 {
                return 1.0;
            }
            // Renting fewer cars than what's on the lot.
            if x < n {
                return rent_dist.pmf(u64::from(x));
            }
            // Renting all cars on lot. Using 1 - CDF ensures probabilities sum to 1.0.
            if x == n {
                return 1.0 - rent_dist.cdf(u64::from(n - 1))
            }
            // x > n scenario is impossible.
            0.0
        }

        fn calc_return_probs(mean: f32, max_n: u8) -> ndarray::Array2<f64> {
            let return_dist = Poisson::new(f64::from(mean)).unwrap();
            let dim = (max_n + 1) as usize;
            let mut y_probs =
                ndarray::Array2::<f64>::zeros((dim, dim));
            for n in 0..max_n + 1 {
                for x in 0..max_n + 1 {
                    y_probs[[n as usize, x as usize]] = 
                        CarProbs::return_prob(n, x, max_n, &return_dist);
                }
            }
            y_probs
        }

        fn return_prob(n :u8, y: u8, max_n: u8, return_dist: &Poisson) -> f64 {
            // Can't fit more than max_n cars on lot.
            if n > max_n {
                return 0.0;
            }
            // Not enough room on lot to return that many cars.
            if y > max_n - n {
                return 0.0;
            }
            // Can only return 0 cars if lot is full.
            if n == max_n {
                return if y == 0 {
                    1.0
                } else {
                    0.0
                }
            }
            // Returning fewer cars than empty spaces on lot.
            if y < max_n - n {
                return return_dist.pmf(u64::from(y));
            }
            // Filliing the lot. Using 1 - CDF ensures probabilities sum to 1.0.
            if y == max_n - n {
                return 1.0 - return_dist.cdf(u64::from(max_n - n - 1));
            }
            0.0
        }

        /// Calculate number of cars rented from the reward and action.
        pub fn cars_rented(r: i16, a: i16) -> u8 {
            if !((r + 2 * a.abs()) % 10 == 0) {
                panic!("Invalid reward for given action.")
            }
            let cars = (r + 2 * a.abs()) / 10;
            if cars > u8::MAX as i16 {
                panic!("Number of cars rented ({}) exceeds maximum.", cars)
            }
            cars as u8
        }

        /// Calculate the reward give the number of cars rented and action.
        pub fn reward(xt: u8, a: i16) -> i16 {
            xt as i16 * 10 - 2 * a.abs()
        }
    }


    #[derive(Debug)]
    pub struct State {
        pub n1: u8,
        pub n2: u8,
    }


    #[derive(Debug)]
    pub struct StateTransitionSolution {
        pub x1: i16,  // Cars rented at site #1
        pub x2: i16,  // Cars rented at site #2
        pub y1: i16,  // Cars returned at site #1
        pub y2: i16,  // Cars returned at site #2
    }

    impl StateTransitionSolution {
        pub fn new(x1: i16, x2: i16) -> StateTransitionSolution {
            let sol =
                StateTransitionSolution { x1, x2,  y1: 0, y2: 0};
            sol
        }
        /// Calculate number of cars rented from reward and number of cars moved (action).


        /// All solution parameters must be non-negative.
        pub fn is_valid(&self) -> bool {
            self.x1 >= 0 && self.x2 >= 0 && self.y1 >= 0 && self.y2 >= 0
        }

        /// Can't rent more cars than what's on lot.
        pub fn is_within_inventory(&self, s1: &State, a: i16) -> bool {
            self.x1 <= s1.n1 as i16 - a && self.x2 <= s1.n2 as i16 + a
        }
        /// Find all possible solutions for transitioning between two states.
        ///
        /// s1 is the initial number of cars at both sites.
        /// s2 is the number of cars at both sites after moving a cars between
        ///     sites. When `a` is positive, a cars are moved from site 1 to site 2.
        ///     When `a` is negative, -a cares are moved from site 2 to site 1.
        ///
        /// There can be as many solutions as the total number of cars rented from
        /// both sites, plus 1.
        pub fn solve(s1: &State, s2: &State, xt: u16, a: i16
        ) -> Vec<StateTransitionSolution> {
            let mut solutions: Vec<StateTransitionSolution> = Vec::new();
            // Can't move more cars than are on site.
            if a > s1.n1 as i16 ||  -a > s1.n2 as i16 {
                return solutions;
            }
            // let total_cars_rented = Solution::cars_rented(r, a);
            for x in 0..xt + 1 {
                let mut z =
                    StateTransitionSolution::new(x as i16, (xt - x) as i16);
                z.y1 = s2.n1 as i16 - s1.n1 as i16 + z.x1 + a;
                z.y2 = s2.n2 as i16 - s1.n2 as i16 + z.x2 - a;
                if z.is_valid() && z.is_within_inventory(s1, a) {
                    solutions.push(z);
                } 
            }
            solutions
        }




    }


    #[cfg(test)]
    mod tests {
        use super::*;
        use core::f32;
        use approx::assert_abs_diff_eq;
        use test_case::test_case;
        
    
        #[test]
        fn rent_probs_small() {
            // Act
            let cprobs = CarProbs::new(
                3, 1.0, 1.0, 2, 1.5, 0.5, 1);
            for px in cprobs.x1.sum_axis(ndarray::Axis(1)) {
                assert_abs_diff_eq!(px as f32, 1.0, epsilon = f32::EPSILON)
            }
            for py in cprobs.y1.sum_axis(ndarray::Axis(1)) {
                assert_abs_diff_eq!(py as f32, 1.0, epsilon = f32::EPSILON)
            }
        }
    
        #[test]
        fn rent_probs_big() {
            // Act
            let cprobs = CarProbs::new(
                20, 3.0, 3.0, 20, 4.0, 2.0, 5);
            for px in cprobs.x1.sum_axis(ndarray::Axis(1)) {
                assert_abs_diff_eq!(px as f32, 1.0, epsilon = f32::EPSILON)
            }
            for py in cprobs.y1.sum_axis(ndarray::Axis(1)) {
                assert_abs_diff_eq!(py as f32, 1.0, epsilon = f32::EPSILON)
            }
        }
    
        #[test_case(0, 0, 0; "Zero rentals and no action")]
        #[test_case(0, 2, -4; "Zero rentals and action from 1 to 2")]
        #[test_case(0, -2, -4; "Zero rentals and action from 2 to 1")]
        #[test_case(5, 0, 50; "Rentals and no action")]
        #[test_case(5, -2, 46; "Rentals and action from 2 to 1")]
        #[test_case(4, 3, 34; "Rentals and action from 1 to 2")]
        fn test_reward_calculation(xt: u8, a: i16, r: i16) {
            assert_eq!(CarProbs::reward(xt, a), r);
        }

        #[test_case(-2, 1, 0; "Negative reward")]
        #[test_case(8, -1, 1; "One car rented")]
        #[test_case(40, 0, 4; "No action")]
        #[test_case(20, 5, 3; "Several cars rented with action")]
        fn test_cars_rented_calculation(r: i16, a: i16, xt: u8) {
            assert_eq!(CarProbs::cars_rented(r, a), xt);
        }
    
    
    }

}

