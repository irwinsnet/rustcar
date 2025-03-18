
use std::cmp;
use statrs::distribution::{Discrete, DiscreteCDF, Poisson};
use crate::policy;
use crate::solver::{State, Outcome, StateIterator};

// use statrs::statistics::Data;

/// Car rental and return probabilities.
/// 
/// Precalculate rental and return probabilities when object is constructed.
/// Indices to probability tables x1, y1, x2, and y2 are
/// [cars on lot, number of cars rented or returned].
/// Use Poisson distribution to calculate probabilities.
pub struct CarProbs {
    /// Maximum number of cars that can be stored at location #1
    pub max1: u8,
    /// Expected number of cars rented each day
    pub rent_mean1: f32,
    /// Expected number of cars returned each day
    pub return_mean1: f32,
    /// Loc 1 rental probs. Indexes: number of cars on lot, number of cars rented
    pub x1: ndarray::Array2<f64>,
    /// Loc 1 return probs. Indexes: number of cars on lot, number of cars returned
    pub y1: ndarray::Array2<f64>,
    /// Maximum number of cars that can be stored at location #2
    pub max2: u8,
    /// Expected number of cars rented each day
    pub rent_mean2: f32,
    /// Expected number of cars returned each day
    pub return_mean2: f32,
    /// Loc 2 rental probs. Indexes: number of cars on lot, number of cars rented
    pub x2: ndarray::Array2<f64>, 
    /// Loc 2 return probs. Indexes: number of cars on lot, number of cars returned
    pub y2: ndarray::Array2<f64>,
    /// Maximum number of cars that can be moved between loc #1 and loc #2
    pub max_move: u8,
    /// Discount rate
    pub g: f64,
    /// Current policy
    pub pi: policy::Policy
}

impl CarProbs {
    pub fn new(
        max1: u8, rent_mean1: f32, return_mean1: f32,
        max2: u8, rent_mean2: f32, return_mean2: f32,
        max_move: u8,
    ) -> CarProbs {
        if max_move > cmp::min(max1, max2) / 2 {
            panic!("Max move must be less than half of smallest lot max.")
        }
        let x1_probs =
            CarProbs::calc_rent_probs(rent_mean1, max1);
        let y1_probs = 
            CarProbs::calc_return_probs(return_mean1, max1);
        let x2_probs =
            CarProbs::calc_rent_probs(rent_mean2, max2);
        let y2_probs = 
            CarProbs::calc_return_probs(return_mean2, max2);
        let pi = policy::Policy::new(max1, max2, max_move);

        CarProbs {
            max1, rent_mean1, return_mean1,
            x1: x1_probs, y1: y1_probs,
            max2, rent_mean2, return_mean2,
            x2: x2_probs, y2: y2_probs,
            max_move,
            g: 0.9, pi
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

    pub fn outcome_prob(&self, s: &State, outcome: &Outcome) -> f64 {
        let p_x1 = self.x1[[s.n1 as usize, outcome.x1 as usize]];
        let p_y1 = self.y1[[s.n1 as usize, outcome.y1 as usize]];
        let p_x2 = self.x2[[s.n2 as usize, outcome.x2 as usize]];
        let p_y2 = self.y2[[s.n2 as usize, outcome.y2 as usize]];
        p_x1 * p_y1 * p_x2 * p_y2
    }

    fn show_array(arr: &ndarray::Array2<f64>, row_prefix: String) {
        print!("    cars on lot:");
        for n in 0..arr.dim().0 {
            print!("{:9}", n);
        }
        println!();
        let mut x = 0;
        for col in arr.columns() {
            print!("{row_prefix}: {x:>3} | ");
            x += 1;
            for elem in col.iter() {
                print!("{:8.4} ", elem);
            }
            println!("");
        }
    }

    pub fn show_probs(&self) {
        println!("\n=== Location #1 Rental Probabilities ===");
        CarProbs::show_array(&self.x1, String::from("  cars rented"));
        println!("\n=== Location #1 Return Probabilities ===");
        CarProbs::show_array(&self.y1, String::from("cars returned"));
        println!("\n=== Location #2 Rental Probabilities ===");
        CarProbs::show_array(&self.x2, String::from("  cars rented"));
        println!("\n=== Location #2 Return Probabilities ===");
        CarProbs::show_array(&self.y2, String::from("cars returned"));
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
    pub fn reward(xt: u32, a: i8) -> i32 {
        xt as i32  * 10 - 2 * a.abs() as i32
    }

    pub fn calc_value(&self, s1: &State) -> f64 {
        let a = self.pi.policy[[s1.n1 as usize, s1.n2 as usize]];
        // Action is invalid if there are not enough cars to move or move exceeds max
        if a > 0 {
            if a + s1.n2 as i8 > self.max2 as i8 || s1.n1 as i8 - a < 0 {
                return 0.0;
            }
        } else if a < 0 {
            if s1.n1 as i8 - a > self.max1 as i8 || s1.n2 as i8 + a < 0 {
                return 0.0;
            } 
        }
        let mut value = 0.0;
        for s2 in StateIterator::new(self.max1, self.max2) {
            let mut v_s2 = self.pi.get_value(s2.n1, s2.n2, a);
            let max_rented = s1.n1.checked_add(s1.n2)
                .expect("Overflow") as u32;
            for xt in 0..(max_rented + 1) {
                let r = CarProbs::reward(xt, a);
                let outcomes = Outcome::solve(s1, &s2, xt, a);
                let mut reward_prob = 0.0;
                for outcome in outcomes {
                    reward_prob += self.outcome_prob(&s2, &outcome);
                }
                value += reward_prob * (r as f64 + self.g * v_s2);
            }
        }
        value
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
    fn test_reward_calculation(xt: u32, a: i8, r: i32) {
        assert_eq!(CarProbs::reward(xt, a), r);
    }

    #[test_case(-2, 1, 0; "Negative reward")]
    #[test_case(8, -1, 1; "One car rented")]
    #[test_case(40, 0, 4; "No action")]
    #[test_case(20, 5, 3; "Several cars rented with action")]
    fn test_cars_rented_calculation(r: i16, a: i16, xt: u8) {
        assert_eq!(CarProbs::cars_rented(r, a), xt);
    }

    #[test]
    fn test_calc_value_no_cars() {
        // Arrange
        let cprobs = CarProbs::new(
            5, 2.0, 2.0,
            5, 2.0, 1.0, 2);
        let s1 = State {n1: 0, n2: 0};
        // Act
        let cv = cprobs.calc_value(&s1);
        // Assert
        assert_eq!(cv, 0.0);
    }

    #[test]
    fn test_calc_value_some_cars() {
        // Arrange
        let cprobs = CarProbs::new(
            5, 2.0, 1.0,
            5, 1.0, 2.0, 2);
        let s1 = State {n1: 1, n2: 1};
        // Act
        let cv = cprobs.calc_value(&s1);
        // Assert
        assert!(cv > 0.0);
        assert!(cv < 20.0);
    }    
}



