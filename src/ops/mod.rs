//! Module of operations that can be performed in a segment tree.
//!
//! A segment tree needs some operation, and this module contains the main
//! [`Operation`] trait, together with the marker trait [`Commutative`] and
//! the trait [`Identity`] for operations with identities. This module also
//! contains implementations for simple operations.
//!
//! Certain methods require inverting operations. The trait
//! [`PartialInvert`] defines a right inverse for an operation, that is, it
//! enables reversing a previously applied operation. The trait [`Invert`]
//! supplies a left inverse, meaning it can create values that can be
//! reversed by the operation.
//!
//! [`Operation`]: trait.Operation.html
//! [`Commutative`]: trait.Commutative.html
//! [`PartialInvert`]: trait.PartialInvert.html
//! [`Invert`]: trait.Invert.html

use std::num::Wrapping;
use std::rc::Rc;
use std::sync::Arc;

/// Implementations for primitive types.
mod primitive;

/// Implementations for types in `num`.
#[cfg(feature = "with-num")]
mod num;

// Traits

/// A trait for an operation over values in one of the crate's structures.
pub trait Operation<N> {
    /// Perform the operation and combine two values together.
    ///
    /// The implementation must ensure that the operation is associative, that
    /// is `combine(combine(a, b), c) = combine(a, combine(b, c))`.
    ///
    /// The other variants exist to allow optimizations through reusing the
    /// arguments. They must all produce the same value.
    fn combine(&self, left: &N, right: &N) -> N;

    /// Performs the operation and stores the result in `left`.
    ///
    /// This must be equivalent to `*left = combine(left, right)` but allows
    /// certain optimizations.
    #[inline]
    fn combine_mut(&self, left: &mut N, right: &N) {
        *left = self.combine(left, right);
    }

    /// Performs the operation and stores the result in `right`.
    ///
    /// This must be equivalent to `*right = combine(left, right)` but allows
    /// certain optimizations.
    #[inline]
    fn combine_mut_right(&self, left: &N, right: &mut N) {
        *right = self.combine(left, right);
    }
}

/// A marker trait for commutative operations.
///
/// The implementation must ensure `combine(a, b) = combine(b, a)`.
pub trait Commutative<N>: Operation<N> {}

/// A trait for operations with identities.
///
/// An identity must satisfy `combine(a, id) = a` and `combine(id, a) = a`.
pub trait Identity<N>: Operation<N> {
    /// Returns any identity.
    fn identity(&self) -> N;
}

/// A trait for operations with a right inverse.
///
/// The implementation must satisfy `invert(combine(a, b), b) = a`.
pub trait PartialInvert<N>: Commutative<N> {
    /// Returns the value `left` such that `combine(left, b) = result`.
    fn invert(&self, result: &N, arg: &N) -> N;

    /// Permform the inversion and stores the result in `result`.
    ///
    /// This must be equivalent to `*result = invert(result, arg)` but allows
    /// certian optimizations.
    #[inline]
    fn invert_mut(&self, result: &mut N, arg: &N) {
        *result = self.invert(result, arg);
    }
}

/// A trait for operations with both a left and a right inverse.
///
/// The implementation must satisfy `combine(invert(a, b), b) = a`.
pub trait Invert<N>: PartialInvert<N> {}

/// A trait for values that can be zero.
pub trait IsZero {
    /// Returns true if `self` is a zero value.
    fn is_zero(&self) -> bool;
}

/// A trait for values that can be NaN.
pub trait IsNan {
    /// Returns true if `self` is a NaN value.
    fn is_nan(&self) -> bool;
}

/// A trait for values that can be finite, but isn't always.
pub trait IsFinite {
    /// Returns true if `self` is a finite value.
    fn is_finite(&self) -> bool;
}

// Wrapper structs

macro_rules! impl_wrapper {
    ($T:ident) => {
        impl<N> $T<N> {
            /// Wraps `value` without checking any constraints.
            #[inline]
            pub fn new_unchecked(value: N) -> $T<N> {
                $T(value)
            }

            /// Consumes `self` and returns the inner value.
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

/// Wrapper that ensures a value isn't zero.
///
/// This can be used for operations that aren't invertible for zero values.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, PartialOrd, Ord)]
pub struct Nonzero<N>(N);

impl<N> Nonzero<N>
    where N: IsZero
{
    /// Wraps `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is zero.
    #[inline]
    pub fn new(value: N) -> Nonzero<N> {
        assert!(!value.is_zero());
        Nonzero(value)
    }
}

impl_wrapper!(Nonzero);

/// Wrapper that ensures a value isn't NaN.
///
/// This can be used for operations that aren't invertible for NaN values.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct NotNan<N>(N);

impl<N> NotNan<N>
    where N: IsNan
{
    /// Wraps `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is NaN.
    #[inline]
    pub fn new(value: N) -> NotNan<N> {
        assert!(!value.is_nan());
        NotNan(value)
    }
}

impl_wrapper!(NotNan);

/// Wrapper that ensures a value is finite.
///
/// This can be used for operations that are only invertible for finite
/// values.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct Finite<N>(N);

impl<N> Finite<N>
    where N: IsFinite
{
    /// Wraps `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` isn't finite.
    #[inline]
    pub fn new(value: N) -> Finite<N> {
        assert!(value.is_finite());
        Finite(value)
    }
}

impl_wrapper!(Finite);

/// Wrapper that ensures a value is finite and nonzero.
///
/// This can be used for operations that are only invertible for finite
/// nonzero values.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, PartialOrd, Ord)]
pub struct FiniteNonzero<N>(N);

impl<N> FiniteNonzero<N>
    where N: IsFinite + IsZero
{
    /// Wraps `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` isn't finite or if `value` is zero.
    #[inline]
    pub fn new(value: N) -> FiniteNonzero<N> {
        assert!(value.is_finite());
        assert!(!value.is_zero());
        FiniteNonzero(value)
    }
}

impl_wrapper!(FiniteNonzero);

// Operations

/// The result is the sum of the arguments.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Add;

/// The result is the sum of the arguments, wrapping on overflow.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct WrappingAdd;

/// The result is the sum of the arguments, saturating on overflow.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct SaturatingAdd;

/// The result is the sum of the arguments, or `None` on overflow.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct CheckedAdd;

/// The result is the product of the arguments.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Mul;

/// The result is the product of the arguments, wrapping on overflow.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct WrappingMul;

/// The result is the product of the arguments, saturating on overflow.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct SaturatingMul;

/// The result is the product of the arguments, or `None` on overflow.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct CheckedMul;

/// The result is the bitwise xor of the arguments.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Xor;

/// The result is the bitwise and of the arguments.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct And;

/// The result is the bitwise or of the arguments.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Or;

/// The result is the maximum value.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Max;

/// The result is the minimum value.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Min;

/// The result is the maximum value, ignoring NaN values.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MaxIgnoreNan;

/// The result is the maximum value, or NaN if any argument is NaN.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MaxTakeNan;

/// The result is the minimum value, ignoring NaN values.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MinIgnoreNan;

/// The result is the minimum value, or NaN if any argument is NaN.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct MinTakeNan;

/// Perform two operations in the same tree.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct Pair<A, B>(pub A, pub B);

/// Adds an identity to an operation by wrapping the type in [`Option`].
///
/// The identity becomes `None` such that `combine(None, None) = None`,
/// `combine(None, Some(a)) = combine(Some(a), None) = Some(a)`.
///
/// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, Hash)]
pub struct WithIdentity<A>(pub A);

// Implementations

impl<'a, N, O> Operation<N> for &'a O
    where O: Operation<N>
{
    #[inline]
    fn combine(&self, left: &N, right: &N) -> N {
        (**self).combine(left, right)
    }

    #[inline]
    fn combine_mut(&self, left: &mut N, right: &N) {
        (**self).combine_mut(left, right)
    }

    #[inline]
    fn combine_mut_right(&self, left: &N, right: &mut N) {
        (**self).combine_mut_right(left, right)
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
    fn invert(&self, result: &N, arg: &N) -> N {
        (**self).invert(result, arg)
    }

    #[inline]
    fn invert_mut(&self, result: &mut N, arg: &N) {
        (**self).invert_mut(result, arg);
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
            fn combine_mut(&self, left: &mut N, right: &N) {
                (**self).combine_mut(left, right);
            }

            #[inline]
            fn combine_mut_right(&self, left: &N, right: &mut N) {
                (**self).combine_mut_right(left, right);
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
            fn invert(&self, result: &N, arg: &N) -> N {
                (**self).invert(result, arg)
            }

            #[inline]
            fn invert_mut(&self, result: &mut N, arg: &N) {
                (**self).invert_mut(result, arg);
            }
        }

        impl<N, $O> Invert<N> for $B where $O: Invert<N> {}
    };
}

impl_op_deref!(O, Box<O>);
impl_op_deref!(O, Rc<O>);
impl_op_deref!(O, Arc<O>);

impl<N, M, A, B> Operation<(N, M)> for Pair<A, B>
    where A: Operation<N>,
          B: Operation<M>
{
    #[inline]
    fn combine(&self, left: &(N, M), right: &(N, M)) -> (N, M) {
        (self.0.combine(&left.0, &right.0), self.1.combine(&left.1, &right.1))
    }

    #[inline]
    fn combine_mut(&self, left: &mut (N, M), right: &(N, M)) {
        self.0.combine_mut(&mut left.0, &right.0);
        self.1.combine_mut(&mut left.1, &right.1);
    }

    #[inline]
    fn combine_mut_right(&self, left: &(N, M), right: &mut (N, M)) {
        self.0.combine_mut_right(&left.0, &mut right.0);
        self.1.combine_mut_right(&left.1, &mut right.1);
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
    #[inline]
    fn identity(&self) -> (N, M) {
        (self.0.identity(), self.1.identity())
    }
}

impl<N, M, A, B> PartialInvert<(N, M)> for Pair<A, B>
    where A: PartialInvert<N>,
          B: PartialInvert<M>
{
    #[inline]
    fn invert(&self, result: &(N, M), arg: &(N, M)) -> (N, M) {
        (self.0.invert(&result.0, &arg.0), self.1.invert(&result.1, &arg.1))
    }

    #[inline]
    fn invert_mut(&self, result: &mut (N, M), arg: &(N, M)) {
        self.0.invert_mut(&mut result.0, &arg.0);
        self.1.invert_mut(&mut result.1, &arg.1);
    }
}

impl<N, M, A, B> Invert<(N, M)> for Pair<A, B>
    where A: Invert<N>,
          B: Invert<M>
{
}

impl<N, O> Operation<Option<N>> for WithIdentity<O>
    where N: Clone,
          O: Operation<N>
{
    #[inline]
    fn combine(&self, left: &Option<N>, right: &Option<N>) -> Option<N> {
        let left_value = match *left {
            Some(ref left) => left,
            None => return right.clone(),
        };

        let right_value = match *right {
            Some(ref right) => right,
            None => return left.clone(),
        };

        Some(self.0.combine(left_value, right_value))
    }

    #[inline]
    fn combine_mut(&self, left: &mut Option<N>, right: &Option<N>) {
        let right = match *right {
            Some(ref right) => right,
            None => return,
        };

        if let Some(ref mut left) = *left {
            self.0.combine_mut(left, right);
            return;
        }

        *left = Some(right.clone());
    }

    #[inline]
    fn combine_mut_right(&self, left: &Option<N>, right: &mut Option<N>) {
        let left_value = match *left {
            Some(ref left) => left,
            None => return,
        };

        if let Some(ref mut right) = *right {
            self.0.combine_mut_right(left_value, right);
            return;
        }

        *right = Some(left_value.clone());
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
    fn invert(&self, result: &Option<N>, arg: &Option<N>) -> Option<N> {
        let arg = match *arg {
            Some(ref arg) => arg,
            None => return result.clone(),
        };

        let result = match *result {
            Some(ref result) => result,
            None => panic!("empty result with nonempty argument"),
        };

        Some(self.0.invert(result, arg))
    }

    #[inline]
    fn invert_mut(&self, result: &mut Option<N>, arg: &Option<N>) {
        let arg = match *arg {
            Some(ref arg) => arg,
            None => return,
        };

        if let Some(ref mut result) = *result {
            self.0.invert_mut(result, arg);
            return;
        }

        panic!("empty result with nonempty argument");
    }
}
