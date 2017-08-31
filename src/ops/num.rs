//! This module requires the `with-num` feature and supplies implementations for various types from
//! the `num` crate.

use std::ops::{Add as OpAdd, Sub, Mul as OpMul, Div};

use num::{BigInt, BigUint, Complex, Zero, One, Signed, Integer};
use num::rational::Ratio;

use ops::{Operation, Commutative, Identity, PartialInvert, Invert, IsNonzero, Nonzero, Add, Mul};

use std::mem;

impl IsNonzero for BigInt {
    #[inline]
    fn is_nonzero(&self) -> bool {
        !self.is_zero()
    }
}

impl IsNonzero for BigUint {
    #[inline]
    fn is_nonzero(&self) -> bool {
        !self.is_zero()
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
            fn combine_left_mut(&self, left: &mut Nonzero<$N>, right: &Nonzero<$N>) {
                // This is intentionally garbage; any mem::replace means
                // we aren't unwind safe in any case.
                let tmp = mem::replace(left, Nonzero::new_unchecked(Zero::zero())).into_inner();
                let right = right.as_ref();

                *left = Nonzero::new(tmp.mul(right));
            }

            #[inline]
            fn combine_right_mut(&self, left: &Nonzero<$N>, right: &mut Nonzero<$N>) {
                // This is intentionally garbage; any mem::replace means
                // we aren't unwind safe in any case.
                let tmp = mem::replace(right, Nonzero::new_unchecked(Zero::zero())).into_inner();
                let left = left.as_ref();

                *right = Nonzero::new(left.mul(tmp));
            }

            #[inline]
            fn combine_left(&self, left: Nonzero<$N>, right: &Nonzero<$N>) -> Nonzero<$N> {
                let left = left.into_inner();
                let right = right.as_ref();

                Nonzero::new(left.mul(right))
            }

            #[inline]
            fn combine_right(&self, left: &Nonzero<$N>, right: Nonzero<$N>) -> Nonzero<$N> {
                let left = left.as_ref();
                let right = right.into_inner();

                Nonzero::new(left.mul(right))
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
            fn invert_mut(&self, result: &mut Nonzero<$N>, arg: &Nonzero<$N>) {
                // This is intentionally garbage; any mem::replace means
                // we aren't unwind safe in any case.
                let tmp = mem::replace(result, Nonzero::new_unchecked(Zero::zero())).into_inner();
                let arg = arg.as_ref();

                *result = Nonzero::new(tmp.div(arg));
            }
        }
    };
}

impl_big_nonzero!(BigUint);
impl_big_nonzero!(BigInt);

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

impl<N> IsNonzero for Ratio<N>
    where N: Clone + Integer
{
    #[inline]
    fn is_nonzero(&self) -> bool {
        !self.is_zero()
    }
}

impl_ratio_op!(Add, add, Zero::zero());

impl<N> PartialInvert<Ratio<N>> for Add
    where N: Clone + Integer
{
    #[inline]
    fn invert_mut(&self, result: &mut Ratio<N>, arg: &Ratio<N>) {
        *result = (&&*result).sub(arg);
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
    fn invert_mut(&self, result: &mut Nonzero<Ratio<N>>, arg: &Nonzero<Ratio<N>>) {
        *result = {
            let tmp = result.as_ref();
            let arg = arg.as_ref();

            Nonzero::new(tmp.div(arg))
        };
    }
}

impl<N> Invert<Nonzero<Ratio<N>>> for Mul where N: Clone + Integer {}
