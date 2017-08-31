//! This module requires the `with-num` feature and supplies implementations for various types from
//! the `num` crate.

use std::ops::{Add as OpAdd, Sub, Mul as OpMul, Div};

use num::{BigInt, BigUint, Complex, Zero, One, Signed, Integer};
use num::rational::Ratio;

use ops::{Operation, Commutative, Identity, PartialInvert, Invert, Add, Mul};

use std::mem;

macro_rules! impl_big_op {
    ($O:ty, $N:ty, $op:ident, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                left.clone().$op(right)
            }

            #[inline]
            fn combine_left_mut(&self, left: &mut $N, right: &$N) {
                let tmp = mem::replace(left, Zero::zero());
                *left = tmp.$op(right);
            }

            #[inline]
            fn combine_right_mut(&self, left: &$N, right: &mut $N) {
                let tmp = mem::replace(right, Zero::zero());
                *right = left.$op(tmp);
            }

            #[inline]
            fn combine_left(&self, left: $N, right: &$N) -> $N {
                left.$op(right)
            }

            #[inline]
            fn combine_right(&self, left: &$N, right: $N) -> $N {
                left.$op(right)
            }
        }

        impl Commutative<$N> for $O {}

        impl Identity<$N> for $O {
            #[inline]
            fn identity(&self) -> $N {
                $identity
            }
        }
    };
}

macro_rules! impl_big_op_partial_inv {
    ($O:ty, $N:ty, $op:ident, $inv:ident, $identity:expr) => {
        impl_big_op!($O, $N, $op, $identity);

        impl PartialInvert<$N> for $O {
            #[inline]
            fn invert_mut(&self, result: &mut $N, arg: &$N) {
                let tmp = mem::replace(result, Zero::zero());
                *result = tmp.$inv(arg);
            }
        }
    };
}

macro_rules! impl_big_op_inv {
    ($O:ty, $N:ty, $op:ident, $inv:ident, $identity:expr) => {
        impl_big_op_partial_inv!($O, $N, $op, $inv, $identity);

        impl Invert<$N> for $O {}
    };
}

impl_big_op_partial_inv!(Add, BigUint, add, sub, Zero::zero());
impl_big_op!(Mul, BigUint, mul, One::one());

impl_big_op_inv!(Add, BigInt, add, sub, Zero::zero());
impl_big_op!(Mul, BigInt, mul, One::one());

impl<N> Operation<Complex<N>> for Add
    where Add: Operation<N>
{
    #[inline]
    fn combine(&self, left: &Complex<N>, right: &Complex<N>) -> Complex<N> {
        Complex {
            re: self.combine(&left.re, &right.re),
            im: self.combine(&left.im, &right.im),
        }
    }

    #[inline]
    fn combine_left_mut(&self, left: &mut Complex<N>, right: &Complex<N>) {
        self.combine_left_mut(&mut left.re, &right.re);
        self.combine_left_mut(&mut left.im, &right.im);
    }

    #[inline]
    fn combine_right_mut(&self, left: &Complex<N>, right: &mut Complex<N>) {
        self.combine_right_mut(&left.re, &mut right.re);
        self.combine_right_mut(&left.im, &mut right.im);
    }

    #[inline]
    fn combine_left(&self, left: Complex<N>, right: &Complex<N>) -> Complex<N> {
        Complex {
            re: self.combine_left(left.re, &right.re),
            im: self.combine_left(left.im, &right.im),
        }
    }

    #[inline]
    fn combine_right(&self, left: &Complex<N>, right: Complex<N>) -> Complex<N> {
        Complex {
            re: self.combine_right(&left.re, right.re),
            im: self.combine_right(&left.im, right.im),
        }
    }

    #[inline]
    fn combine_both(&self, left: Complex<N>, right: Complex<N>) -> Complex<N> {
        Complex {
            re: self.combine_both(left.re, right.re),
            im: self.combine_both(left.im, right.im),
        }
    }
}

impl<N> Commutative<Complex<N>> for Add where Add: Commutative<N> {}

impl<N> Identity<Complex<N>> for Add
    where Add: Identity<N>
{
    #[inline]
    fn identity(&self) -> Complex<N> {
        Complex {
            re: self.identity(),
            im: self.identity(),
        }
    }
}

impl<N> PartialInvert<Complex<N>> for Add
    where Add: PartialInvert<N>
{
    #[inline]
    fn invert_mut(&self, result: &mut Complex<N>, arg: &Complex<N>) {
        self.invert_mut(&mut result.re, &arg.re);
        self.invert_mut(&mut result.im, &arg.im);
    }

    #[inline]
    fn invert(&self, result: Complex<N>, arg: &Complex<N>) -> Complex<N> {
        Complex {
            re: self.invert(result.re, &arg.re),
            im: self.invert(result.im, &arg.im),
        }
    }
}

macro_rules! impl_ratio_op_partial_inv {
    ($O:ty, $op:ident, $inv:ident, $identity:expr) => {
        // It turns out that `Ratio` always clones its arguments, so there's no
        // gain in using the specialized versions of the methods.

        impl<N> Operation<Ratio<N>> for $O
            where N: Clone + Integer
        {
            #[inline]
            fn combine(&self, left: &Ratio<N>, right: &Ratio<N>) -> Ratio<N> {
                left.$op(right)
            }
        }

        impl<N> Commutative<Ratio<N>> for $O where N: Clone + Integer {}

        impl<N> Identity<Ratio<N>> for $O
            where N: Clone + Integer
        {
            #[inline]
            fn identity(&self) -> Ratio<N> {
                $identity
            }
        }

        impl<N> PartialInvert<Ratio<N>> for $O
            where N: Clone + Integer
        {
            #[inline]
            fn invert_mut(&self, result: &mut Ratio<N>, arg: &Ratio<N>) {
                *result = (&&*result).$inv(arg);
            }
        }
    };
}

macro_rules! impl_ratio_op_inv {
    ($O:ty, $op:ident, $inv:ident, $identity:expr) => {
        impl_ratio_op_partial_inv!($O, $op, $inv, $identity);

        impl<N> Invert<Ratio<N>> for $O where N: Clone + Integer + Signed {}
    };
}

impl_ratio_op_inv!(Add, add, sub, One::one());
impl_ratio_op_partial_inv!(Mul, mul, div, Zero::zero());

// Mul is not Invert<Ratio<N>> even when N: Signed, since
// multiplication by 0 isn't invertible.
