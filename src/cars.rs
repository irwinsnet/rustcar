
use std::{cmp, io};
use statrs::distribution::{Discrete, DiscreteCDF, Poisson};
use crate::policy;
use crate::solver::{State, Outcome, StateIterator};

// use statrs::statistics::Data;


#[derive(Debug)]
pub struct OutcomeProb {
    pub s1_n1: u8,
    pub s1_n2: u8,
    pub s2_n1: u8,
    pub s2_n2: u8,
    pub xt: u32,
    pub a: i8,
    pub r: i32,
    pub x1: i32,
    pub x2: i32,
    pub y1: i32,
    pub y2: i32,
    pub prob: f64
}

impl OutcomeProb {
    pub fn new(
        s1: &State, s2: &State, xt: u32, a: i8, r: i32, ocome: &Outcome, prob: f64
    ) -> OutcomeProb {
        OutcomeProb {
            s1_n1: s1.n1, s1_n2: s1.n2, s2_n1: s2.n1, s2_n2: s2.n2,
            xt, a, r,
            x1: ocome.x1, x2: ocome.x2, y1: ocome.y1, y2: ocome.y2,
            prob
        }
    }
}


/// Car rental and return probabilities.
/// 
/// Specify the maximum number of cars allowed at each location and the
/// maximum number of cars that can be moved when initializing the struct.
/// 
/// Precalculate rental and return probabilities when object is constructed.
/// Indices to probability tables x1, y1, x2, and y2 are
/// [cars on lot, number of cars rented or returned].
/// Use Poisson distribution to calculate probabilities.
pub struct RentalAgency {
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
}

impl RentalAgency {

    /// Create a new struct with pre-calculated probabilities.
    pub fn new(
        max1: u8, rent_mean1: f32, return_mean1: f32,
        max2: u8, rent_mean2: f32, return_mean2: f32,
        max_move: u8,
    ) -> RentalAgency {
        if max_move > cmp::min(max1, max2) / 2 {
            panic!("Max move must be less than half of smallest lot max.")
        }
        let x1_probs =
            RentalAgency::calc_rent_probs(rent_mean1, max1);
        let y1_probs = 
            RentalAgency::calc_return_probs(return_mean1, max1);
        let x2_probs =
            RentalAgency::calc_rent_probs(rent_mean2, max2);
        let y2_probs = 
            RentalAgency::calc_return_probs(return_mean2, max2);

        RentalAgency {
            max1, rent_mean1, return_mean1,
            x1: x1_probs, y1: y1_probs,
            max2, rent_mean2, return_mean2,
            x2: x2_probs, y2: y2_probs,
            max_move,
            g: 0.9
        }
    }

    /// Calculate the rental probabilities from the mean and max car limit.
    /// 
    /// The mean is the mean number of cars that are rented each day.
    /// Obviously you can't rent more cars than what's on the lot, so for
    /// x = max_n, p(x) = 1 - P(x-1) where P is cumulative Poisson
    /// distribution.
    fn calc_rent_probs(mean: f32, max_n: u8) -> ndarray::Array2<f64> {
        let rent_dist = Poisson::new(f64::from(mean)).unwrap();
        let dim = (max_n + 1) as usize;
        let mut x_probs =
            ndarray::Array2::<f64>::zeros((dim, dim));
        for n in 0..max_n + 1 {
            for x in 0..max_n + 1 {
                x_probs[[n as usize, x as usize]] = 
                    RentalAgency::rent_prob(n, x, max_n, &rent_dist);
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

    /// Calculate the return probabilities from the mean and max car limit.
    /// 
    /// The mean is the mean number of cars that are returned each day.
    /// Obviously you can't return more cars than what can fit on the lot, so
    /// for y = max_n - n, p(y) = 1 - P(y-1) where P is cumulative Poisson
    /// distribution.
    fn calc_return_probs(mean: f32, max_n: u8) -> ndarray::Array2<f64> {
        let return_dist = Poisson::new(f64::from(mean)).unwrap();
        let dim = (max_n + 1) as usize;
        let mut y_probs =
            ndarray::Array2::<f64>::zeros((dim, dim));
        for n in 0..max_n + 1 {
            for y in 0..max_n + 1 {
                y_probs[[n as usize, y as usize]] = 
                    RentalAgency::return_prob(n, y, max_n, &return_dist);
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

    /// Calculate the probability for set of rental and return totals.
    /// 
    /// Assumes that site #1 and site #2 rental and return probabilities are
    /// indepdendent. Probabilities depend on number of care rented or returned
    /// and the number of cars on the lot.
    pub fn outcome_prob(&self, s: &State, a: i8, outcome: &Outcome) -> f64 {
        let n1 = (s.n1 as i8 - a) as usize;
        let p_x1 = self.x1[[n1, outcome.x1 as usize]];
        let p_y1 = self.y1[[n1 - outcome.x1 as usize, outcome.y1 as usize]];
        let n2 = (s.n2 as i8 + a) as usize;
        let p_x2 = self.x2[[n2, outcome.x2 as usize]];
        let p_y2 = self.y2[[n2 - outcome.x2 as usize,outcome.y2 as usize]];
        p_x1 * p_y1 * p_x2 * p_y2
    }

    /// Display a probability table on the command line, for troubleshooting.
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

    /// Show all four probability tables in the terminal.
    pub fn show_probs(&self) {
        println!("\n=== Location #1 Rental Probabilities ===");
        RentalAgency::show_array(&self.x1, String::from("  cars rented"));
        println!("\n=== Location #1 Return Probabilities ===");
        RentalAgency::show_array(&self.y1, String::from("cars returned"));
        println!("\n=== Location #2 Rental Probabilities ===");
        RentalAgency::show_array(&self.x2, String::from("  cars rented"));
        println!("\n=== Location #2 Return Probabilities ===");
        RentalAgency::show_array(&self.y2, String::from("cars returned"));
    }

    /// Send probability table to standard out, as CSV text.
    pub fn array_to_csv(arr: &ndarray::Array2<f64>) {
        let mut wtr = csv::Writer::from_writer(io::stdout());
        for i in 0..arr.dim().0 {
            for j in 0..arr.dim().1 {
                wtr.write_field(format!("{:.4}", arr[[i, j]]));
            }
            wtr.write_record(None::<&[u8]>);
        }

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

    /// Calculate the reward given the number of cars rented and action.
    pub fn reward(xt: u32, a: i8) -> i32 {
        xt as i32  * 10 - 2 * a.abs() as i32
    }

    /// Calculate value for a given state, assume action is per current policy.
    ///
    /// The state is the number of cars at site #1 and site #2 at the beginning
    /// of the turn. The value is the discounted, expected total reward.
    // pub fn calc_value(&self, s1: &State) -> f64 {
    //     let a = self.pi.policy[[s1.n1 as usize, s1.n2 as usize]];
    //     self.calc_value_for_action(s1, a)
    // }

    /// Calculate the value for a given state and action.
    /// 
    /// The action need not be per the current policy.
    /// 
    /// Iterate over all possible states and rewards. Calculate the probability
    /// of each state-reward combination and multiply it times the sum of the
    /// expected reward and the discounted values of the next state (s2).
    pub fn calc_value_for_action(
        &self, s1: &State, a: i8, pi: &policy::Policy) -> f64 {
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
            let mut v_s2 = pi.get_value(s2.n1, s2.n2, a);
            let max_rented = s1.n1.checked_add(s1.n2)
                .expect("Overflow") as u32;
            for xt in 0..(max_rented + 1) {
                let (r, reward_prob, _) = self.calc_reward_prob(s1, &s2, a, xt);
                value += reward_prob * (r as f64 + self.g * v_s2);
            }
        }
        value
    }

    /// Calculate probability of state s2 with reward r, given state s1 and action a.
    pub fn calc_reward_prob(
        &self, s1: &State, s2: &State, a: i8, xt: u32
    ) -> (i32, f64, Vec<OutcomeProb>)  {
        let r = RentalAgency::reward(xt, a);
        let outcomes = Outcome::solve(s1, &s2, xt, a);
        let mut reward_prob = 0.0;
        let mut oprobs: Vec<OutcomeProb> = Vec::new();
        for outcome in outcomes {
            let prob = self.outcome_prob(&s1, a, &outcome);
            oprobs.push(
                OutcomeProb::new(s1, s2, xt, a, r, &outcome, prob)
            );
            reward_prob += prob;
        }
        (r, reward_prob, oprobs)
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
        let cprobs = RentalAgency::new(
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
        let cprobs = RentalAgency::new(
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
        assert_eq!(RentalAgency::reward(xt, a), r);
    }

    #[test_case(-2, 1, 0; "Negative reward")]
    #[test_case(8, -1, 1; "One car rented")]
    #[test_case(40, 0, 4; "No action")]
    #[test_case(20, 5, 3; "Several cars rented with action")]
    fn test_cars_rented_calculation(r: i16, a: i16, xt: u8) {
        assert_eq!(RentalAgency::cars_rented(r, a), xt);
    }

    #[test]
    fn test_calc_value_no_cars() {
        // Arrange
        let cprobs = RentalAgency::new(
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
        let cprobs = RentalAgency::new(
            5, 2.0, 1.0,
            5, 1.0, 2.0, 2);
        let s1 = State {n1: 1, n2: 1};
        // Act
        let cv = cprobs.calc_value(&s1);
        // Assert
        assert!(cv > 0.0);
        assert!(cv < 20.0);
    }

    #[test]
    fn test_scenario1() {
        // Arrange
        let cprobs = RentalAgency::new(
            3, 2.0, 1.0, 3, 1.0, 2.0, 1);
        let s1 = State { n1: 1, n2: 1 };
        let s2 = State { n1: 0, n2: 0 };
        // Act
        let (r, prob, trace) = cprobs.calc_reward_prob(&s1, &s2, 0, 2);
        for ocome in trace {
            println!("{:?}", ocome);
        }
    }

    #[test]
    fn view_array() {
        let cprobs = RentalAgency::new(
            3, 2.0, 1.0, 3, 1.0, 2.0, 1);
            RentalAgency::array_to_csv(&cprobs.y2);
    }


}



