//! This module requires the `with-num` feature and supplies implementations for various types from
//! the `num` crate.

use std::mem;
use std::ops::{Add as OpAdd, Sub, Mul as OpMul, Div};

use num::{BigInt, BigUint, Complex, Zero, One, Signed, Integer};
use num::rational::Ratio;

use super::*;

// Bignums

impl IsZero for BigInt {
    #[inline]
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
}

impl IsZero for BigUint {
    #[inline]
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
}

macro_rules! impl_big_op {
    ($O:ty, $N:ty, $op:ident, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                left.clone().$op(right)
            }

            #[inline]
            fn combine_mut(&self, left: &mut $N, right: &$N) {
                let tmp = mem::replace(left, Default::default());
                *left = tmp.$op(right);
            }

            #[inline]
            fn combine_mut_right(&self, left: &$N, right: &mut $N) {
                let tmp = mem::replace(right, Default::default());
                *right = left.$op(tmp);
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
            fn invert(&self, result: &$N, arg: &$N) -> $N {
                result.clone().$inv(arg)
            }

            #[inline]
            fn invert_mut(&self, result: &mut $N, arg: &$N) {
                let tmp = mem::replace(result, Default::default());
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

macro_rules! impl_big_nonzero {
    ($N:ty) => {
        impl Operation<Nonzero<$N>> for Mul {
            #[inline]
            fn combine(&self, left: &Nonzero<$N>, right: &Nonzero<$N>) -> Nonzero<$N> {
                let left = left.as_ref().clone();
                let right = right.as_ref();

                Nonzero::new(left.mul(right))
            }

            #[inline]
            fn combine_mut(&self, left: &mut Nonzero<$N>, right: &Nonzero<$N>) {
                // This invalid value is okay here, since we don't try to be unwind safe.
                let garbage = Nonzero::new_unchecked(Default::default());

                let tmp = mem::replace(left, garbage).into_inner();
                let right = right.as_ref();

                *left = Nonzero::new(tmp.mul(right));
            }

            #[inline]
            fn combine_mut_right(&self, left: &Nonzero<$N>, right: &mut Nonzero<$N>) {
                // This invalid value is okay here, since we don't try to be unwind safe.
                let garbage = Nonzero::new_unchecked(Default::default());

                let tmp = mem::replace(right, garbage).into_inner();
                let left = left.as_ref();

                *right = Nonzero::new(left.mul(tmp));
            }
        }

        impl Commutative<Nonzero<$N>> for Mul {}

        impl Identity<Nonzero<$N>> for Mul {
            #[inline]
            fn identity(&self) -> Nonzero<$N> {
                Nonzero::new_unchecked(One::one())
            }
        }

        impl PartialInvert<Nonzero<$N>> for Mul {
            #[inline]
            fn invert(&self, result: &Nonzero<$N>, arg: &Nonzero<$N>) -> Nonzero<$N> {
                let result = result.as_ref().clone();
                let arg = arg.as_ref();

                Nonzero::new(result.div(arg))
            }

            #[inline]
            fn invert_mut(&self, result: &mut Nonzero<$N>, arg: &Nonzero<$N>) {
                // This invalid value is okay here, since we don't try to be unwind safe.
                let garbage = Nonzero::new_unchecked(Default::default());

                let tmp = mem::replace(result, garbage).into_inner();
                let arg = arg.as_ref();

                *result = Nonzero::new(tmp.div(arg));
            }
        }
    };
}

impl_big_nonzero!(BigUint);
impl_big_nonzero!(BigInt);

impl Identity<BigUint> for Max {
    #[inline]
    fn identity(&self) -> BigUint {
        Zero::zero()
    }
}

// Complex

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
    fn combine_mut(&self, left: &mut Complex<N>, right: &Complex<N>) {
        self.combine_mut(&mut left.re, &right.re);
        self.combine_mut(&mut left.im, &right.im);
    }

    #[inline]
    fn combine_mut_right(&self, left: &Complex<N>, right: &mut Complex<N>) {
        self.combine_mut_right(&left.re, &mut right.re);
        self.combine_mut_right(&left.im, &mut right.im);
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
    fn invert(&self, result: &Complex<N>, arg: &Complex<N>) -> Complex<N> {
        Complex {
            re: self.invert(&result.re, &arg.re),
            im: self.invert(&result.im, &arg.im),
        }
    }

    #[inline]
    fn invert_mut(&self, result: &mut Complex<N>, arg: &Complex<N>) {
        self.invert_mut(&mut result.re, &arg.re);
        self.invert_mut(&mut result.im, &arg.im);
    }
}

// Ratio

// It turns out that `Ratio` always clones its arguments, so there's no
// gain in using the specialized versions of the methods.

macro_rules! impl_ratio_op {
    ($O:ty, $op:ident, $identity:expr) => {
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
    };
}

impl<N> IsZero for Ratio<N>
    where N: Clone + Integer
{
    #[inline]
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
}

impl_ratio_op!(Add, add, Zero::zero());

impl<N> PartialInvert<Ratio<N>> for Add
    where N: Clone + Integer
{
    #[inline]
    fn invert(&self, result: &Ratio<N>, arg: &Ratio<N>) -> Ratio<N> {
        result.sub(arg)
    }
}

impl<N> Invert<Ratio<N>> for Add where N: Clone + Integer + Signed {}

impl_ratio_op!(Mul, mul, One::one());

impl<N> Operation<Nonzero<Ratio<N>>> for Mul
    where N: Clone + Integer
{
    #[inline]
    fn combine(&self, left: &Nonzero<Ratio<N>>, right: &Nonzero<Ratio<N>>) -> Nonzero<Ratio<N>> {
        let left = left.as_ref();
        let right = right.as_ref();

        Nonzero::new(left.mul(right))
    }
}

impl<N> Commutative<Nonzero<Ratio<N>>> for Mul where N: Clone + Integer {}

impl<N> Identity<Nonzero<Ratio<N>>> for Mul
    where N: Clone + Integer
{
    #[inline]
    fn identity(&self) -> Nonzero<Ratio<N>> {
        Nonzero::new_unchecked(One::one())
    }
}

impl<N> PartialInvert<Nonzero<Ratio<N>>> for Mul
    where N: Clone + Integer
{
    #[inline]
    fn invert(&self, result: &Nonzero<Ratio<N>>, arg: &Nonzero<Ratio<N>>) -> Nonzero<Ratio<N>> {
        let result = result.as_ref();
        let arg = arg.as_ref();

        Nonzero::new(result.div(arg))
    }
}

impl<N> Invert<Nonzero<Ratio<N>>> for Mul where N: Clone + Integer {}
