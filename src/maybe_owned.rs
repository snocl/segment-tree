//! This module contains a variant of [`Cow`] that doesn't require [`Clone`].
//!
//! [`Cow`]: https://doc.rust-lang.org/std/borrow/enum.Cow.html
//! [`Clone`]: https://doc.rust-lang.org/std/clone/trait.Clone.html

use std::cmp::{self, Ordering};
use std::default::Default;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// A variant of [`Cow`] that doesn't require [`Clone`].
///
/// [`Cow`]: https://doc.rust-lang.org/std/borrow/enum.Cow.html
/// [`Clone`]: https://doc.rust-lang.org/std/clone/trait.Clone.html
#[derive(Clone, Debug)]
pub enum MaybeOwned<'a, T: 'a> {
    Borrowed(&'a T),
    Owned(T),
}

impl<'a, T: 'a> MaybeOwned<'a, T> {
    /// Get a reference to the contained value.
    #[inline]
    pub fn borrow(&self) -> &T {
        match *self {
            MaybeOwned::Borrowed(v) => v,
            MaybeOwned::Owned(ref v) => v,
        }
    }
}

impl<'a, T: 'a + Clone> MaybeOwned<'a, T> {
    /// Get the value, cloning if necessary.
    #[inline]
    pub fn unwrap_or_clone(self) -> T {
        match self {
            MaybeOwned::Borrowed(v) => v.clone(),
            MaybeOwned::Owned(v) => v,
        }
    }
}

impl<'a, T: 'a + PartialEq> PartialEq for MaybeOwned<'a, T> {
    #[inline]
    fn eq(&self, other: &MaybeOwned<T>) -> bool {
        self.borrow().eq(other.borrow())
    }

    #[inline]
    fn ne(&self, other: &MaybeOwned<T>) -> bool {
        self.borrow().ne(other.borrow())
    }
}
impl<'a, T: 'a + Eq> cmp::Eq for MaybeOwned<'a, T> {}

impl<'a, T: 'a + PartialOrd> cmp::PartialOrd for MaybeOwned<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &MaybeOwned<T>) -> Option<cmp::Ordering> {
        self.borrow().partial_cmp(other.borrow())
    }

    #[inline]
    fn lt(&self, other: &MaybeOwned<T>) -> bool {
        self.borrow().lt(other.borrow())
    }

    #[inline]
    fn le(&self, other: &MaybeOwned<T>) -> bool {
        self.borrow().le(other.borrow())
    }

    #[inline]
    fn gt(&self, other: &MaybeOwned<T>) -> bool {
        self.borrow().gt(other.borrow())
    }

    #[inline]
    fn ge(&self, other: &MaybeOwned<T>) -> bool {
        self.borrow().ge(other.borrow())
    }
}

impl<'a, T: 'a + Display> Display for MaybeOwned<'a, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

impl<'a, T: 'a + Ord> Ord for MaybeOwned<'a, T> {
    #[inline]
    fn cmp(&self, other: &MaybeOwned<T>) -> Ordering {
        self.borrow().cmp(other.borrow())
    }
}

impl<'a, T: 'a + Default> Default for MaybeOwned<'a, T> {
    #[inline]
    fn default() -> MaybeOwned<'a, T> {
        MaybeOwned::Owned(Default::default())
    }
}

impl<'a, T: 'a + Hash> Hash for MaybeOwned<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().hash(state)
    }
}
