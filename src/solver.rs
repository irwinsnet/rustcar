#![allow(unused)]

pub mod solver {
    use std::cmp;
    use std::fmt;
    use std::iter::Iterator;
    use crate::cars;

    #[derive(Debug, Eq, PartialEq, Hash)]
    pub struct State {
        pub n1: u8,  // Number of cars at site #1 at start of day
        pub n2: u8,  // Number of cars at site #2 at start of day
    }
    impl fmt::Display for State {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "(n1: {}, n2: {})", self.n1, self.n2)
        }
    }

    struct StateIterator {
        n1: u8,
        n2: u8,
        max_n1: u8,
        max_n2: u8,
    }
    impl StateIterator {
        pub fn new(max_n1: u8, max_n2: u8) -> StateIterator {
            StateIterator {n1: 0, n2: 0, max_n1, max_n2}
        }
    }
    impl Iterator for StateIterator {
        type Item = State;

        fn next<'a>(&mut self) -> Option<Self::Item> {
            if self.n1 > self.max_n1 {
                return None;
            }
            let state = State{n1: self.n1, n2: self.n2};
            if self.n2 < self.max_n2 {
                self.n2 += 1;
            } else {
                self.n2 = 0;
                self.n1 += 1;
            }

            return Some(state);
        }
    }


    /// The number of cars rented and returned at each site during a day.
    #[derive(Debug, Eq, PartialEq)]
    pub struct Outcome {
        pub x1: i16,  // Cars rented at site #1
        pub x2: i16,  // Cars rented at site #2
        pub y1: i16,  // Cars returned at site #1
        pub y2: i16,  // Cars returned at site #2
    }

    impl fmt::Display for Outcome {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "(x1: {}, y1: {}, x2: {} y2: {})",
                   self.x1, self.y1, self.x2, self.y2)
        }
    }

    impl Outcome {
        pub fn new(x1: i16, x2: i16) -> Outcome {
            let outcome = Outcome { x1, x2,  y1: 0, y2: 0 };
            outcome
        }

        fn is_nonnegative(&self) -> bool {
            self.x1 >= 0 && self.x2 >= 0 && self.y1 >= 0 && self.y2 >= 0
        }

        fn locations_have_enough_cars(&self, s1: &State, a: i16) -> bool {
            let loc1_enough_cars = self.x1 <= s1.n1 as i16 - a;
            let loc2_enough_cars = self.x2 <= s1.n2 as i16 + a;
            loc1_enough_cars && loc2_enough_cars
        }

        /// Find all outcomes that transition from state s1 to state s2
        /// 
        /// The `solve` function checks for the following error conditions:
        /// * Attempting to move more cars than what's available on the lot
        /// * More cars are rented than what's on the lot
        /// * the number of retruned cars can't be negative
        /// 
        /// Does NOT verify that the action won't cause the maximum number of
        /// cars on a lot to be exceeded. For example, if the maximum number of
        /// cars allowed on lot #1 is five, and there are four cars on lot #1,
        /// and we attempt to move 2 cars to lot #1 from lot #2, `solve` will
        /// allow the maximum number of cars on the lot to be exceeded. This is
        /// because `State` does not know the maximum number of cars allowed at
        /// each location. The options for dealing with this are:
        /// * Add max car limits to the `State` struct and use those fields
        ///   `solve`
        /// * Have the calling code verify action a is valid.
        pub fn solve(s1: &State, s2: &State, xt: u16, a: i16) -> Vec<Outcome> {
            let mut outcomes: Vec<Outcome> = Vec::new();
            // Can't move more cars than what's on lot
            if a > s1.n1 as i16 ||  -a > s1.n2 as i16 {
                return outcomes;
            }
            // Can't rent more cars than what's on lot.
            if xt > (s1.n1 + s1.n2) as u16 {
                return outcomes;
            }
            for x in 0..(xt + 1) {
                let mut z = Outcome::new(x as i16, (xt - x) as i16);
                if !z.locations_have_enough_cars(s1, a) {
                    continue
                }
                z.y1 = s2.n1 as i16 - s1.n1 as i16 + z.x1 + a;
                z.y2 = s2.n2 as i16 - s1.n2 as i16 + z.x2 - a;
                if z.is_nonnegative() {
                    outcomes.push(z);
                }
            }
            outcomes
        }
    }

    #[cfg(test)]
    mod tests {
        use std::collections::HashSet;

        use super::*;

        #[test]
        fn iterate_states() {
            // Arrange
            let state_iter = StateIterator::new(2, 2);
            let mut states: HashSet<State> = HashSet::new();
            // Act
            for s in state_iter {
                assert!(s.n1 <= 2);
                assert!(s.n2 <= 2);
                states.insert(s);
            }
            // Assert
            assert_eq!(states.len(), 9);
        }

        fn check_outcome(
            s1: &State, s2: &State, outcome: &Outcome, xt: u16, a: i16
        ) -> bool {
            let site1_valid = 
                s1.n1 as i16 - a - outcome.x1 + outcome.y1 == s2.n1 as i16;
            let site2_valid = 
                s1.n2 as i16 + a - outcome.x2 + outcome.y2 == s2.n2 as i16;
            let sufficient_cars =
                xt <= s1.n1 as u16 + s1.n2 as u16;
            site1_valid && site2_valid && sufficient_cars
        }

        #[test]
        fn solve_for_outcome_zero_states_and_rentals() {
            // Arrange
            let s1 = State {n1: 0, n2: 0};
            let s2 = State {n1: 0, n2: 0};
            // Act
            let outcomes = Outcome::solve(&s1, &s2, 0, 0);
            // Assert
            assert_eq!(outcomes.len(), 1 as usize);
            assert_eq!(outcomes[0], Outcome {x1: 0, y1: 0, x2: 0, y2: 0});
        }

        #[test]
        fn solve_for_outcome_zero_states_nonzero_rentals() {
            // Arrange
            let s1 = State {n1: 3, n2: 3};
            let s2 = State {n1: 3, n2: 3};
            let xt: u16 = 3;
            let a: i16 = 0;

            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            // Assert
            assert_eq!(outcomes.len(), 4 as usize);
            assert_eq!(outcomes[0], Outcome {x1: 0, y1: 0, x2: 3, y2: 3});
            assert_eq!(outcomes[outcomes.len() - 1], Outcome {x1: 3, y1: 3, x2: 0, y2: 0});
            for outcome in outcomes {
                assert!(check_outcome(&s1, &s2, &outcome, xt, a));
            }
        }
    
        #[test]
        fn solve_for_outcome_zero_states_nonzero_rentals_action() {
            // Arrange
            let s1 = State {n1: 1, n2: 1};
            let s2 = State {n1: 1, n2: 1};
            let xt: u16 = 2;
            let a: i16 = 0;
            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            // Assert
            assert_eq!(outcomes.len(), 3 as usize);
            assert_eq!(outcomes[0], Outcome {x1: 0, y1: 0, x2: 2, y2: 2});
            assert_eq!(outcomes[outcomes.len() - 1], Outcome {x1: 2, y1: 2, x2: 0, y2: 0});
            for outcome in outcomes {
                assert!(check_outcome(&s1, &s2, &outcome, xt, a));
            }
        }

        #[test]
        fn solve_for_outcome_not_enough_cars() {
            // Arrange
            let s1 = State {n1: 1, n2: 1};
            let s2 = State {n1: 1, n2: 1};
            let xt: u16 = 3;
            let a: i16 = 0;
            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            // Assert
            assert_eq!(outcomes.len(), 0 as usize);
        }

        #[test]
        fn solve_for_outcome_dont_exceed_site_inventory() {
            // Arrange
            let s1 = State {n1: 2, n2: 2};
            let s2 = State {n1: 2, n2: 2};
            let xt: u16 = 3;
            let a: i16 = 0;
            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            // Assert
            // for outcome in outcomes {
            //     println!("{}", outcome);
            // }
            assert_eq!(outcomes.len(), 2 as usize);
            assert_eq!(outcomes[0], Outcome {x1: 1, y1: 1, x2: 2, y2: 2});
            assert_eq!(outcomes[outcomes.len() - 1], Outcome {x1: 2, y1: 2, x2: 1, y2: 1});
            for outcome in outcomes {
                assert!(check_outcome(&s1, &s2, &outcome, xt, a));
            }
        }

        #[test]
        fn solve_for_outcome_with_move() {
            // Arrange
            let s1 = State {n1: 2, n2: 2};
            let s2 = State {n1: 2, n2: 2};
            let xt: u16 = 3;
            let a: i16 = 1;
            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            assert_eq!(outcomes.len(), 2 as usize);
            assert_eq!(outcomes[0], Outcome {x1: 0, y1: 1, x2: 3, y2: 2});
            assert_eq!(outcomes[outcomes.len() - 1], Outcome {x1: 1, y1: 2, x2: 2, y2: 1});
            for outcome in outcomes {
                assert!(check_outcome(&s1, &s2, &outcome, xt, a));
            }
        }

        #[test]
        fn solve_for_outcome_with_negative_move() {
            // Arrange
            let s1 = State {n1: 2, n2: 2};
            let s2 = State {n1: 2, n2: 2};
            let xt: u16 = 3;
            let a: i16 = -2;
            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            // Assert
            assert_eq!(outcomes.len(), 1 as usize);
            assert_eq!(outcomes[0], Outcome {x1: 3, y1: 1, x2: 0, y2: 2});
            for outcome in outcomes {
                assert!(check_outcome(&s1, &s2, &outcome, xt, a));
            }
        }

        #[test]
        fn solve_for_outcome_no_negative_returns() {
            // Arrange
            let s1 = State {n1: 5, n2: 5};
            let s2 = State {n1: 0, n2: 5};
            let xt: u16 = 2;
            let a: i16 = 0;
            // Act
            let outcomes = Outcome::solve(&s1, &s2, xt, a);
            // Assert
            assert_eq!(outcomes.len(), 0 as usize);
        }

/*
    Modifications required for solver:
    * Doesn't prevent negative returns or rentals.

*/



    }

}
