//! This module provides an alternative to [`Cow`] that doesn't require [`Clone`].
//!
//! [`Cow`]: https://doc.rust-lang.org/std/borrow/enum.Cow.html
//! [`Clone`]: https://doc.rust-lang.org/std/clone/trait.Clone.html

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

use self::MaybeOwned::{Borrowed, Owned};

/// An alternative to [`Cow`] that doesn't require [`Clone`].
///
/// [`Cow`]: https://doc.rust-lang.org/std/borrow/enum.Cow.html
/// [`Clone`]: https://doc.rust-lang.org/std/clone/trait.Clone.html
#[derive(Clone, Debug)]
pub enum MaybeOwned<'a, T: 'a> {
    Borrowed(&'a T),
    Owned(T),
}

impl<'a, T: 'a> From<T> for MaybeOwned<'a, T> {
    #[inline]
    fn from(value: T) -> MaybeOwned<'a, T> {
        MaybeOwned::Owned(value)
    }
}

impl<'a, T: 'a> From<&'a T> for MaybeOwned<'a, T> {
    #[inline]
    fn from(value: &'a T) -> MaybeOwned<'a, T> {
        MaybeOwned::Borrowed(value)
    }
}

impl<'a, T: 'a> MaybeOwned<'a, T>
    where T: Clone
{
    /// Returns an owned version of the value, cloning if necessary.
    #[inline]
    pub fn into_owned(self) -> T {
        match self {
            Borrowed(value) => value.clone(),
            Owned(value) => value,
        }
    }
}

impl<'a, T: 'a> AsRef<T> for MaybeOwned<'a, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        match *self {
            Borrowed(value) => value,
            Owned(ref value) => value,
        }
    }
}

impl<'a, T: 'a> PartialEq for MaybeOwned<'a, T>
    where T: PartialEq
{
    #[inline]
    fn eq(&self, other: &MaybeOwned<T>) -> bool {
        self.as_ref() == other.as_ref()
    }

    #[inline]
    fn ne(&self, other: &MaybeOwned<T>) -> bool {
        self.as_ref() != other.as_ref()
    }
}
impl<'a, T: 'a> Eq for MaybeOwned<'a, T> where T: Eq {}

impl<'a, T: 'a + PartialOrd> PartialOrd for MaybeOwned<'a, T>
    where T: PartialOrd
{
    #[inline]
    fn partial_cmp(&self, other: &MaybeOwned<'a, T>) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }

    #[inline]
    fn lt(&self, other: &MaybeOwned<T>) -> bool {
        self.as_ref().lt(other.as_ref())
    }

    #[inline]
    fn le(&self, other: &MaybeOwned<T>) -> bool {
        self.as_ref().le(other.as_ref())
    }

    #[inline]
    fn gt(&self, other: &MaybeOwned<T>) -> bool {
        self.as_ref().gt(other.as_ref())
    }

    #[inline]
    fn ge(&self, other: &MaybeOwned<T>) -> bool {
        self.as_ref().ge(other.as_ref())
    }
}

impl<'a, T: 'a> Display for MaybeOwned<'a, T>
    where T: Display
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl<'a, T: 'a> Ord for MaybeOwned<'a, T>
    where T: Ord
{
    #[inline]
    fn cmp(&self, other: &MaybeOwned<T>) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<'a, T: 'a> Default for MaybeOwned<'a, T>
    where T: Default
{
    #[inline]
    fn default() -> MaybeOwned<'a, T> {
        MaybeOwned::Owned(Default::default())
    }
}

impl<'a, T: 'a + Hash> Hash for MaybeOwned<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}
