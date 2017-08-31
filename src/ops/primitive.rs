use std::cmp;
use std::{u8, u16, u32, u64, i8, i16, i32, i64, usize, isize, f32, f64};
use std::ops::{Add as OpAdd, Sub, Mul as OpMul, Div, BitXor, BitAnd, BitOr};

use super::*;

// General implementation macros

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
            fn invert(&self, result: &$N, arg: &$N) -> $N {
                result.$inv(*arg)
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
            fn invert(&self, result: &$N, arg: &$N) -> $N {
                $init(result.into_inner().$inv(arg.into_inner()))
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

/// Integer implementation macros

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

macro_rules! impl_primitive_int {
    ($N:ident) => {
        impl IsZero for $N {
            #[inline]
            fn is_zero(&self) -> bool {
                *self == 0
            }
        }

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

        impl_primitive_op_cmp!(Max, $N, max, $N::MIN);
        impl_primitive_op_cmp!(Min, $N, min, $N::MAX);
    };
}

macro_rules! impl_primitive_unsigned_int {
    ($N:ident) => {
        impl_primitive_int!($N);

        impl_primitive_op_partial_inv!(Add, $N, add, sub, 0);
    };
}

macro_rules! impl_primitive_signed_int {
    ($N:ident) => {
        impl_primitive_int!($N);

        impl_primitive_op_inv!(Add, $N, add, sub, 0);
    };
}

// Floating-point implementation macro

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

macro_rules! impl_primitive_float {
    ($N:ident) => {
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

        impl_primitive_op_cmp_deref!(Max, NotNan<$N>, gt, NotNan::new_unchecked($N::NEG_INFINITY));
        impl_primitive_op_cmp_deref!(Min, NotNan<$N>, lt, NotNan::new_unchecked($N::INFINITY));

        impl_primitive_op_cmp_nan!(MaxIgnoreNan, $N, gt, false, $N::NEG_INFINITY);
        impl_primitive_op_cmp_nan!(MaxTakeNan, $N, gt, true, $N::NEG_INFINITY);
        impl_primitive_op_cmp_nan!(MinIgnoreNan, $N, lt, false, $N::INFINITY);
        impl_primitive_op_cmp_nan!(MinTakeNan, $N, lt, true, $N::INFINITY);
    };
}

// Implementations

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

impl_primitive_float!(f32);
impl_primitive_float!(f64);

// Tests

#[cfg(test)]
mod tests {
    use std::{f32, i32};

    use super::*;

    #[test]
    fn ops_nan() {
        assert_eq!(1.0, MaxIgnoreNan.combine(&0.0, &1.0));
        assert_eq!(1.0, MaxIgnoreNan.combine(&1.0, &0.0));
        assert_eq!(1.0, MaxIgnoreNan.combine(&f32::NAN, &1.0));
        assert_eq!(1.0, MaxIgnoreNan.combine(&1.0, &f32::NAN));
        assert_eq!(f32::NEG_INFINITY,
                   MaxIgnoreNan.combine(&f32::NAN, &f32::NEG_INFINITY));
        assert_eq!(f32::NEG_INFINITY,
                   MaxIgnoreNan.combine(&f32::NEG_INFINITY, &f32::NAN));
        assert!(MaxIgnoreNan.combine(&f32::NAN, &f32::NAN).is_nan());

        assert_eq!(0.0, MinIgnoreNan.combine(&0.0, &1.0));
        assert_eq!(0.0, MinIgnoreNan.combine(&1.0, &0.0));
        assert_eq!(1.0, MinIgnoreNan.combine(&f32::NAN, &1.0));
        assert_eq!(1.0, MinIgnoreNan.combine(&1.0, &f32::NAN));
        assert_eq!(f32::INFINITY,
                   MinIgnoreNan.combine(&f32::NAN, &f32::INFINITY));
        assert_eq!(f32::INFINITY,
                   MinIgnoreNan.combine(&f32::INFINITY, &f32::NAN));
        assert!(MinIgnoreNan.combine(&f32::NAN, &f32::NAN).is_nan());

        assert_eq!(1.0, MaxTakeNan.combine(&0.0, &1.0));
        assert_eq!(1.0, MaxTakeNan.combine(&1.0, &0.0));
        assert!(MaxTakeNan.combine(&f32::NAN, &f32::INFINITY).is_nan());
        assert!(MaxTakeNan.combine(&f32::INFINITY, &f32::NAN).is_nan());
        assert!(MaxTakeNan.combine(&f32::NAN, &f32::NEG_INFINITY).is_nan());
        assert!(MaxTakeNan.combine(&f32::NEG_INFINITY, &f32::NAN).is_nan());
        assert!(MaxTakeNan.combine(&f32::NAN, &f32::NAN).is_nan());

        assert_eq!(0.0, MinTakeNan.combine(&0.0, &1.0));
        assert_eq!(0.0, MinTakeNan.combine(&1.0, &0.0));
        assert!(MinTakeNan.combine(&f32::NAN, &f32::INFINITY).is_nan());
        assert!(MinTakeNan.combine(&f32::INFINITY, &f32::NAN).is_nan());
        assert!(MinTakeNan.combine(&f32::NAN, &f32::NEG_INFINITY).is_nan());
        assert!(MinTakeNan.combine(&f32::NEG_INFINITY, &f32::NAN).is_nan());
        assert!(MinTakeNan.combine(&f32::NAN, &f32::NAN).is_nan());
    }

    #[test]
    fn ops_and_identity() {
        for i in -200i32..201i32 {
            assert_eq!(And.combine(&i, &And.identity()), i);
        }
        assert_eq!(i32::MAX, And.combine(&i32::MAX, &And.identity()));
        assert_eq!(i32::MIN, And.combine(&i32::MIN, &And.identity()));
    }
}