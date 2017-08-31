//! Module of operations that can be performed in a segment tree.
//!
//! A segment tree needs some operation, and this module contains the main [`Operation`] trait,
//! together with the marker trait [`Commutative`]. This module also contains
//! implementations for simple operations.
//!
//! [`Operation`]: trait.Operation.html
//! [`Commutative`]: trait.Commutative.html

use std::cmp;
use std::num::Wrapping;
use std::ops::{Add as OpAdd, Sub, Mul as OpMul, Div, BitXor, BitAnd, BitOr};
use std::rc::Rc;
use std::sync::Arc;
use std::{u8, u16, u32, u64, i8, i16, i32, i64, usize, isize, f32, f64};

/// A trait that specifies which associative operator to use in a segment tree.
pub trait Operation<N> {
    /// The operation that is performed to combine two intervals in the segment tree.
    ///
    /// This function must be associative, that is `combine(combine(a, b), c) = combine(a,
    /// combine(b, c))`.
    fn combine(&self, left: &N, right: &N) -> N;

    /// Replace the value in a with `combine(a, b)`. This function exists to allow certain
    /// optimizations and by default simply calls `combine`.
    #[inline]
    fn combine_left_mut(&self, left: &mut N, right: &N) {
        *left = self.combine(left, right);
    }

    /// Replace the value in a with `combine(a, b)`. This function exists to allow certain
    /// optimizations and by default simply calls `combine`.
    #[inline]
    fn combine_right_mut(&self, left: &N, right: &mut N) {
        *right = self.combine(left, right);
    }

    /// Must return the same as `combine`. This function exists to allow certain optimizations
    /// and by default simply calls `combine_left_mut`.
    #[inline]
    fn combine_left(&self, mut left: N, right: &N) -> N {
        self.combine_left_mut(&mut left, right);
        left
    }

    /// Must return the same as `combine`. This function exists to allow certain optimizations
    /// and by default simply calls `combine_right_mut`.
    #[inline]
    fn combine_right(&self, left: &N, mut right: N) -> N {
        self.combine_right_mut(left, &mut right);
        right
    }

    /// Must return the same as `combine`. This function exists to allow certain optimizations
    /// and by default simply calls `combine_left`.
    #[inline]
    fn combine_both(&self, left: N, right: N) -> N {
        self.combine_left(left, &right)
    }
}

/// A marker trait that specifies that an operation is commutative.
///
/// The implementation must ensure that `combine(a, b) = combine(b, a)` holds.
pub trait Commutative<N>: Operation<N> {}

/// A trait that specifies that this operation has an identity.
///
/// An identity must satisfy `combine(a, id) = a` and `combine(id, a) = a`.
pub trait Identity<N>: Operation<N> {
    /// Returns any identity.
    fn identity(&self) -> N;
}

pub trait PartialInvert<N>: Commutative<N> {
    fn invert_mut(&self, result: &mut N, arg: &N);

    #[inline]
    fn invert(&self, mut result: N, arg: &N) -> N {
        self.invert_mut(&mut result, arg);
        result
    }
}

pub trait Invert<N>: PartialInvert<N> {}

impl<'a, N, O> Operation<N> for &'a O
    where O: Operation<N>
{
    #[inline]
    fn combine(&self, left: &N, right: &N) -> N {
        (**self).combine(left, right)
    }

    #[inline]
    fn combine_left_mut(&self, left: &mut N, right: &N) {
        (**self).combine_left_mut(left, right);
    }

    #[inline]
    fn combine_right_mut(&self, left: &N, right: &mut N) {
        (**self).combine_right_mut(left, right);
    }

    #[inline]
    fn combine_left(&self, left: N, right: &N) -> N {
        (**self).combine_left(left, right)
    }

    #[inline]
    fn combine_right(&self, left: &N, right: N) -> N {
        (**self).combine_right(left, right)
    }

    #[inline]
    fn combine_both(&self, left: N, right: N) -> N {
        (**self).combine_both(left, right)
    }
}

impl<'a, N, O> Commutative<N> for &'a O where O: Commutative<N> {}

impl<'a, N, O> Identity<N> for &'a O
    where O: Identity<N>
{
    #[inline]
    fn identity(&self) -> N {
        (**self).identity()
    }
}

impl<'a, N, O> PartialInvert<N> for &'a O
    where O: PartialInvert<N>
{
    #[inline]
    fn invert_mut(&self, result: &mut N, right: &N) {
        (**self).invert_mut(result, right);
    }

    #[inline]
    fn invert(&self, result: N, right: &N) -> N {
        (**self).invert(result, right)
    }
}

impl<'a, N, O> Invert<N> for &'a O where O: Invert<N> {}

macro_rules! impl_op_deref {
    ($O:ident, $B:ty) => {
        impl<N, $O> Operation<N> for $B where $O: Operation<N> {
            #[inline]
            fn combine(&self, left: &N, right: &N) -> N {
                (**self).combine(left, right)
            }

            #[inline]
            fn combine_left_mut(&self, left: &mut N, right: &N) {
                (**self).combine_left_mut(left, right);
            }

            #[inline]
            fn combine_right_mut(&self, left: &N, right: &mut N) {
                (**self).combine_right_mut(left, right);
            }

            #[inline]
            fn combine_left(&self, left: N, right: &N) -> N {
                (**self).combine_left(left, right)
            }

            #[inline]
            fn combine_right(&self, left: &N, right: N) -> N {
                (**self).combine_right(left, right)
            }

            #[inline]
            fn combine_both(&self, left: N, right: N) -> N {
                (**self).combine_both(left, right)
            }
        }

        impl<N, $O> Commutative<N> for $B where $O: Commutative<N> {}

        impl<N, $O> Identity<N> for $B where $O: Identity<N> {
            #[inline]
            fn identity(&self) -> N {
                (**self).identity()
            }
        }

        impl<N, $O> PartialInvert<N> for $B where $O: PartialInvert<N> {
            #[inline]
            fn invert_mut(&self, result: &mut N, arg: &N) {
                (**self).invert_mut(result, arg);
            }

            #[inline]
            fn invert(&self, result: N, arg: &N) -> N {
                (**self).invert(result, arg)
            }
        }

        impl<N, $O> Invert<N> for $B where $O: Invert<N> {}
    };
}

impl_op_deref!(O, Box<O>);
impl_op_deref!(O, Rc<O>);
impl_op_deref!(O, Arc<O>);

pub trait IsZero {
    fn is_zero(&self) -> bool;
}

macro_rules! impl_is_zero {
    ($N:ty) => {
        impl IsZero for $N {
            #[inline]
            fn is_zero(&self) -> bool {
                *self == 0
            }
        }
    };
}

impl_is_zero!(u8);
impl_is_zero!(u16);
impl_is_zero!(u32);
impl_is_zero!(u64);
impl_is_zero!(i8);
impl_is_zero!(i16);
impl_is_zero!(i32);
impl_is_zero!(i64);
impl_is_zero!(usize);
impl_is_zero!(isize);

macro_rules! impl_wrapper {
    ($T:ident) => {
        impl<N> $T<N> {
            #[inline]
            pub fn new_unchecked(value: N) -> $T<N> {
                $T(value)
            }

            #[inline]
            pub fn into_inner(self) -> N {
                self.0
            }
        }

        impl<N> AsRef<N> for $T<N> {
            fn as_ref(&self) -> &N {
                &self.0
            }
        }
    };
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, PartialOrd, Ord)]
pub struct Nonzero<N>(N);

impl<N> Nonzero<N>
    where N: IsZero
{
    #[inline]
    pub fn new(value: N) -> Nonzero<N> {
        assert!(!value.is_zero());
        Nonzero(value)
    }
}

impl_wrapper!(Nonzero);

/// Each node contains the sum of the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Add;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct WrappingAdd;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct SaturatingAdd;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct CheckedAdd;

/// Each node contains the product of the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Mul;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct WrappingMul;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct SaturatingMul;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct CheckedMul;

/// Each node contains the bitwise xor of the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Xor;

/// Each node contains the bitwise and of the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct And;

/// Each node contains the bitwise or of the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Or;

macro_rules! impl_primitive_op {
    ($O:ty, $N:ty, $op:ident, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                left.$op(*right)
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

macro_rules! impl_primitive_op_partial_inv {
    ($O:ty, $N:ty, $op:ident, $inv:ident, $identity:expr) => {
        impl_primitive_op!($O, $N, $op, $identity);

        impl PartialInvert<$N> for $O {
            #[inline]
            fn invert_mut(&self, result: &mut $N, arg: &$N) {
                *result = result.$inv(*arg);
            }
        }
    };
}

macro_rules! impl_primitive_op_inv {
    ($O:ty, $N:ty, $op:ident, $inv:ident, $identity:expr) => {
        impl_primitive_op_partial_inv!($O, $N, $op, $inv, $identity);

        impl Invert<$N> for $O {}
    };
}

macro_rules! impl_primitive_op_partial_inv_deref {
    ($O:ty, $N:ty, $init:expr, $op:ident, $inv:ident, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                $init(left.into_inner().$op(right.into_inner()))
            }
        }

        impl Commutative<$N> for $O {}

        impl Identity<$N> for $O {
            #[inline]
            fn identity(&self) -> $N {
                $identity
            }
        }

        impl PartialInvert<$N> for $O {
            #[inline]
            fn invert_mut(&self, result: &mut $N, arg: &$N) {
                *result = $init(result.into_inner().$inv(arg.into_inner()));
            }
        }
    };
}

macro_rules! impl_primitive_op_inv_deref {
    ($O:ty, $N:ty, $init:expr, $op:ident, $inv:ident, $identity:expr) => {
        impl_primitive_op_partial_inv_deref!($O, $N, $init, $op, $inv, $identity);

        impl Invert<$N> for $O {}
    };
}

macro_rules! impl_primitive_op_checked {
    ($O:ty, $N:ty, $op:ident, $identity:expr) => {
        impl Operation<Option<$N>> for $O {
            #[inline]
            fn combine(&self, left: &Option<$N>, right: &Option<$N>) -> Option<$N> {
                match (*left, *right) {
                    (Some(ref left), Some(ref right)) => left.$op(*right),
                    _ => None,
                }
            }
        }

        impl Commutative<Option<$N>> for $O {}

        impl Identity<Option<$N>> for $O {
            #[inline]
            fn identity(&self) -> Option<$N> {
                Some($identity)
            }
        }
    };
}

macro_rules! impl_primitive_int {
    ($N:ty) => {
        impl_primitive_op_inv!(WrappingAdd, $N, wrapping_add, wrapping_sub, 0);
        impl_primitive_op!(SaturatingAdd, $N, saturating_add, 0);
        impl_primitive_op!(Mul, $N, mul, 1);
        impl_primitive_op_partial_inv_deref!(Mul, Nonzero<$N>, Nonzero::new, mul, div,
                                             Nonzero::new_unchecked(1));
        impl_primitive_op!(WrappingMul, $N, wrapping_mul, 0);
        impl_primitive_op!(SaturatingMul, $N, saturating_mul, 1);

        impl_primitive_op_checked!(CheckedAdd, $N, checked_add, 0);
        impl_primitive_op_checked!(CheckedMul, $N, checked_mul, 1);

        impl_primitive_op_inv!(Xor, $N, bitxor, bitxor, 0);
        impl_primitive_op!(And, $N, bitand, !0);
        impl_primitive_op!(Or, $N, bitor, 0);

        impl_primitive_op_inv!(Add, Wrapping<$N>, add, sub, Wrapping(0));
        impl_primitive_op!(Mul, Wrapping<$N>, mul, Wrapping(1));
    };
}

macro_rules! impl_primitive_unsigned_int {
    ($N:ty) => {
        impl_primitive_int!($N);

        impl_primitive_op_partial_inv!(Add, $N, add, sub, 0);
    };
}

macro_rules! impl_primitive_signed_int {
    ($N:ty) => {
        impl_primitive_int!($N);

        impl_primitive_op_inv!(Add, $N, add, sub, 0);
    };
}

impl_primitive_unsigned_int!(u8);
impl_primitive_unsigned_int!(u16);
impl_primitive_unsigned_int!(u32);
impl_primitive_unsigned_int!(u64);
impl_primitive_signed_int!(i8);
impl_primitive_signed_int!(i16);
impl_primitive_signed_int!(i32);
impl_primitive_signed_int!(i64);
impl_primitive_unsigned_int!(usize);
impl_primitive_signed_int!(isize);

pub trait IsNan {
    fn is_nan(&self) -> bool;
}

pub trait IsFinite {
    fn is_finite(&self) -> bool;
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct NotNan<N>(N);

impl<N> NotNan<N>
    where N: IsNan
{
    #[inline]
    pub fn new(value: N) -> NotNan<N> {
        assert!(!value.is_nan());
        NotNan(value)
    }
}

impl_wrapper!(NotNan);

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct Finite<N>(N);

impl<N> Finite<N>
    where N: IsFinite
{
    #[inline]
    pub fn new(value: N) -> Finite<N> {
        assert!(value.is_finite());
        Finite(value)
    }
}

impl_wrapper!(Finite);

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, PartialOrd, Ord)]
pub struct FiniteNonzero<N>(N);

impl<N> FiniteNonzero<N>
    where N: IsFinite + IsZero
{
    #[inline]
    pub fn new(value: N) -> FiniteNonzero<N> {
        assert!(value.is_finite());
        assert!(!value.is_zero());
        FiniteNonzero(value)
    }
}

impl_wrapper!(FiniteNonzero);

macro_rules! impl_primitive_float {
    ($N:ty) => {
        impl IsNan for $N {
            #[inline]
            fn is_nan(&self) -> bool {
                (*self).is_nan()
            }
        }

        impl IsFinite for $N {
            #[inline]
            fn is_finite(&self) -> bool {
                (*self).is_finite()
            }
        }

        impl IsZero for $N {
            #[inline]
            fn is_zero(&self) -> bool {
                *self == 0.0
            }
        }

        impl_primitive_op!(Add, $N, add, 0.0);
        impl_primitive_op!(Mul, $N, mul, 1.0);

        impl_primitive_op_inv_deref!(Add, Finite<$N>, Finite::new, add, sub,
                                     Finite::new_unchecked(0.0));
        impl_primitive_op_inv_deref!(Mul, FiniteNonzero<$N>, FiniteNonzero::new, mul, div,
                                     FiniteNonzero::new_unchecked(1.0));
    };
}

impl_primitive_float!(f32);
impl_primitive_float!(f64);

/// Each node contains the maximum value in the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Max;

/// Each node contains the minimum value in the interval it represents.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Min;

macro_rules! impl_primitive_op_cmp {
    ($O:ty, $N:ty, $op:ident, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                cmp::$op(*left, *right)
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

macro_rules! impl_primitive_cmp_int {
    ($N:tt) => {
        impl_primitive_op_cmp!(Max, $N, max, $N::MIN);
        impl_primitive_op_cmp!(Min, $N, min, $N::MAX);
    };
}

impl_primitive_cmp_int!(u8);
impl_primitive_cmp_int!(u16);
impl_primitive_cmp_int!(u32);
impl_primitive_cmp_int!(u64);
impl_primitive_cmp_int!(i8);
impl_primitive_cmp_int!(i16);
impl_primitive_cmp_int!(i32);
impl_primitive_cmp_int!(i64);
impl_primitive_cmp_int!(usize);
impl_primitive_cmp_int!(isize);

/// Variant of [`Max`] that considers NaN smaller than anything.
///
/// [`Max`]: struct.Max.html
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MaxIgnoreNan;

/// Variant of [`Max`] that considers NaN larger than anything.
///
/// [`Max`]: struct.Max.html
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MaxTakeNan;

/// Variant of [`Min`] that considers NaN larger than anything.
///
/// [`Min`]: struct.Min.html
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MinIgnoreNan;

/// Variant of [`Min`] that considers NaN smaller than anything.
///
/// [`Min`]: struct.Min.html
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MinTakeNan;

macro_rules! impl_primitive_op_cmp_nan {
    ($O:ty, $N:ty, $cmp:ident, $take_nan:expr, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                let propagate_left = if $take_nan { left.is_nan() } else { right.is_nan() };
                if propagate_left || left.$cmp(right) { *left } else { *right }
            }
        }

        impl Commutative<$N> for $O {}

        impl Identity<$N> for $O {
            fn identity(&self) -> $N {
                $identity
            }
        }
    };
}

macro_rules! impl_primitive_op_cmp_deref {
    ($O:ty, $N:ty, $cmp:tt, $identity:expr) => {
        impl Operation<$N> for $O {
            #[inline]
            fn combine(&self, left: &$N, right: &$N) -> $N {
                if left.as_ref().$cmp(right.as_ref()) { *left } else { *right }
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

macro_rules! impl_primitive_cmp_float {
    ($N:tt) => {
        impl_primitive_op_cmp_deref!(Max, NotNan<$N>, gt, NotNan::new_unchecked($N::NEG_INFINITY));
        impl_primitive_op_cmp_deref!(Min, NotNan<$N>, lt, NotNan::new_unchecked($N::INFINITY));

        impl_primitive_op_cmp_nan!(MaxIgnoreNan, $N, gt, false, $N::NEG_INFINITY);
        impl_primitive_op_cmp_nan!(MaxTakeNan, $N, gt, true, $N::NEG_INFINITY);
        impl_primitive_op_cmp_nan!(MinIgnoreNan, $N, lt, false, $N::INFINITY);
        impl_primitive_op_cmp_nan!(MinTakeNan, $N, lt, true, $N::INFINITY);
    };
}

impl_primitive_cmp_float!(f32);
impl_primitive_cmp_float!(f64);

#[cfg(test)]
mod tests {
    use std::{f32, i32};
    use super::*;

    #[test]
    fn ops_nan() {
        assert_eq!(MaxIgnoreNan.combine_both(0.0, 1.0), 1.0);
        assert_eq!(MaxIgnoreNan.combine_both(1.0, 0.0), 1.0);
        assert_eq!(MaxIgnoreNan.combine_both(f32::NAN, 1.0), 1.0);
        assert_eq!(MaxIgnoreNan.combine_both(1.0, f32::NAN), 1.0);
        assert_eq!(MaxIgnoreNan.combine_both(f32::NAN, f32::NEG_INFINITY),
                   f32::NEG_INFINITY);
        assert_eq!(MaxIgnoreNan.combine_both(f32::NEG_INFINITY, f32::NAN),
                   f32::NEG_INFINITY);
        assert!(MaxIgnoreNan.combine_both(f32::NAN, f32::NAN).is_nan());

        assert_eq!(MinIgnoreNan.combine_both(0.0, 1.0), 0.0);
        assert_eq!(MinIgnoreNan.combine_both(1.0, 0.0), 0.0);
        assert_eq!(MinIgnoreNan.combine_both(f32::NAN, 1.0), 1.0);
        assert_eq!(MinIgnoreNan.combine_both(1.0, f32::NAN), 1.0);
        assert_eq!(MinIgnoreNan.combine_both(f32::NAN, f32::INFINITY),
                   f32::INFINITY);
        assert_eq!(MinIgnoreNan.combine_both(f32::INFINITY, f32::NAN),
                   f32::INFINITY);
        assert!(MinIgnoreNan.combine_both(f32::NAN, f32::NAN).is_nan());

        assert_eq!(MaxTakeNan.combine_both(0.0, 1.0), 1.0);
        assert_eq!(MaxTakeNan.combine_both(1.0, 0.0), 1.0);
        assert!(MaxTakeNan.combine_both(f32::NAN, f32::INFINITY).is_nan());
        assert!(MaxTakeNan.combine_both(f32::INFINITY, f32::NAN).is_nan());
        assert!(MaxTakeNan.combine_both(f32::NAN, f32::NEG_INFINITY).is_nan());
        assert!(MaxTakeNan.combine_both(f32::NEG_INFINITY, f32::NAN).is_nan());
        assert!(MaxTakeNan.combine_both(f32::NAN, f32::NAN).is_nan());

        assert_eq!(MinTakeNan.combine_both(0.0, 1.0), 0.0);
        assert_eq!(MinTakeNan.combine_both(1.0, 0.0), 0.0);
        assert!(MinTakeNan.combine_both(f32::NAN, f32::INFINITY).is_nan());
        assert!(MinTakeNan.combine_both(f32::INFINITY, f32::NAN).is_nan());
        assert!(MinTakeNan.combine_both(f32::NAN, f32::NEG_INFINITY).is_nan());
        assert!(MinTakeNan.combine_both(f32::NEG_INFINITY, f32::NAN).is_nan());
        assert!(MinTakeNan.combine_both(f32::NAN, f32::NAN).is_nan());
    }

    #[test]
    fn ops_and_identity() {
        for i in -200i32..201i32 {
            assert_eq!(And.combine_both(i, And.identity()), i);
        }
        assert_eq!(And.combine_both(i32::MAX, And.identity()), i32::MAX);
        assert_eq!(And.combine_both(i32::MIN, And.identity()), i32::MIN);
    }
}

/// Store more information in each node.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Pair<A, B>(pub A, pub B);

impl<N, M, A, B> Operation<(N, M)> for Pair<A, B>
    where A: Operation<N>,
          B: Operation<M>
{
    #[inline]
    fn combine(&self, left: &(N, M), right: &(N, M)) -> (N, M) {
        (self.0.combine(&left.0, &right.0), self.1.combine(&left.1, &right.1))
    }

    #[inline]
    fn combine_left_mut(&self, left: &mut (N, M), right: &(N, M)) {
        self.0.combine_left_mut(&mut left.0, &right.0);
        self.1.combine_left_mut(&mut left.1, &right.1);
    }

    #[inline]
    fn combine_right_mut(&self, left: &(N, M), right: &mut (N, M)) {
        self.0.combine_right_mut(&left.0, &mut right.0);
        self.1.combine_right_mut(&left.1, &mut right.1);
    }

    #[inline]
    fn combine_left(&self, left: (N, M), right: &(N, M)) -> (N, M) {
        (self.0.combine_left(left.0, &right.0), self.1.combine_left(left.1, &right.1))
    }

    #[inline]
    fn combine_right(&self, left: &(N, M), right: (N, M)) -> (N, M) {
        (self.0.combine_right(&left.0, right.0), self.1.combine_right(&left.1, right.1))
    }

    #[inline]
    fn combine_both(&self, left: (N, M), right: (N, M)) -> (N, M) {
        (self.0.combine_both(left.0, right.0), self.1.combine_both(left.1, right.1))
    }
}

impl<N, M, A, B> Commutative<(N, M)> for Pair<A, B>
    where A: Commutative<N>,
          B: Commutative<M>
{
}

impl<N, M, A, B> Identity<(N, M)> for Pair<A, B>
    where A: Identity<N>,
          B: Identity<M>
{
    fn identity(&self) -> (N, M) {
        (self.0.identity(), self.1.identity())
    }
}

impl<N, M, A, B> PartialInvert<(N, M)> for Pair<A, B>
    where A: PartialInvert<N>,
          B: PartialInvert<M>
{
    #[inline]
    fn invert_mut(&self, result: &mut (N, M), arg: &(N, M)) {
        self.0.invert_mut(&mut result.0, &arg.0);
        self.1.invert_mut(&mut result.1, &arg.1);
    }

    #[inline]
    fn invert(&self, result: (N, M), arg: &(N, M)) -> (N, M) {
        (self.0.invert(result.0, &arg.0), self.1.invert(result.1, &arg.1))
    }
}

impl<N, M, A, B> Invert<(N, M)> for Pair<A, B>
    where A: Invert<N>,
          B: Invert<M>
{
}

/// Adds an identity to an operation by wrapping the type in [`Option`]. Clones when combined with
/// None.
///
/// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct WithIdentity<A>(pub A);

impl<N, O> Operation<Option<N>> for WithIdentity<O>
    where N: Clone,
          O: Operation<N>
{
    #[inline]
    fn combine(&self, left: &Option<N>, right: &Option<N>) -> Option<N> {
        match (left.as_ref(), right.as_ref()) {
            (left, None) => left.cloned(),
            (None, right) => right.cloned(),
            (Some(left), Some(right)) => Some(self.0.combine(left, right)),
        }
    }

    #[inline]
    fn combine_left_mut(&self, left: &mut Option<N>, right: &Option<N>) {
        match (left.as_mut(), right.as_ref()) {
            (_, None) => return,
            (None, _) => {}
            (Some(left), Some(right)) => {
                self.0.combine_left_mut(left, right);
                return;
            }
        }
        *left = right.clone();
    }

    #[inline]
    fn combine_right_mut(&self, left: &Option<N>, right: &mut Option<N>) {
        match (left.as_ref(), right.as_mut()) {
            (None, _) => return,
            (_, None) => {}
            (Some(left), Some(right)) => {
                self.0.combine_right_mut(left, right);
                return;
            }
        }
        *right = left.clone();
    }

    #[inline]
    fn combine_left(&self, left: Option<N>, right: &Option<N>) -> Option<N> {
        match (left, right.as_ref()) {
            (left, None) => left,
            (None, right) => right.cloned(),
            (Some(left), Some(right)) => Some(self.0.combine_left(left, right)),
        }
    }

    #[inline]
    fn combine_right(&self, left: &Option<N>, right: Option<N>) -> Option<N> {
        match (left.as_ref(), right) {
            (left, None) => left.cloned(),
            (None, right) => right,
            (Some(left), Some(right)) => Some(self.0.combine_right(left, right)),
        }
    }
}

impl<N, O> Commutative<Option<N>> for WithIdentity<O>
    where N: Clone,
          O: Operation<N>
{
}

impl<N, O> Identity<Option<N>> for WithIdentity<O>
    where N: Clone,
          O: Operation<N>
{
    #[inline]
    fn identity(&self) -> Option<N> {
        None
    }
}

impl<N, O> PartialInvert<Option<N>> for WithIdentity<O>
    where N: Clone,
          O: PartialInvert<N>
{
    #[inline]
    fn invert_mut(&self, result: &mut Option<N>, arg: &Option<N>) {
        let arg = match arg.as_ref() {
            Some(arg) => arg,
            None => return,
        };

        match result.as_mut() {
            None => panic!("empty result with nonempty argument"),
            Some(result) => self.0.invert_mut(result, arg),
        }
    }

    #[inline]
    fn invert(&self, result: Option<N>, arg: &Option<N>) -> Option<N> {
        let arg = match arg.as_ref() {
            Some(arg) => arg,
            None => return result,
        };

        match result {
            None => panic!("empty result with nonempty argument"),
            Some(result) => Some(self.0.invert(result, arg)),
        }
    }
}

// We put this down here so that num.rs can use our macros.
#[cfg(feature = "with-num")]
mod num;
