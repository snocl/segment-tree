use std::default::Default;

use maybe_owned::MaybeOwned;
use ops::{Commutative, PartialInvert, Invert};

/// This data structure allows prefix queries and single element modification.
///
/// This tree allocates `n * sizeof(N)` bytes of memory, and can be resized.
///
/// This data structure is implemented using a Fenwick tree, which is also known as a binary
/// indexed tree.
///
/// # Examples
///
/// Showcase of functionality:
///
/// ```rust
/// use segment_tree::ops::Add;
/// use segment_tree::PrefixPoint;
///
/// let buf = vec![10, 5, 30, 40];
///
/// let mut pp = PrefixPoint::build(buf, Add);
///
/// // If we query, we get the sum up until the specified value.
/// assert_eq!(pp.query(0), 10);
/// assert_eq!(pp.query(1), 15);
/// assert_eq!(pp.query(2), 45);
/// assert_eq!(pp.query(3), 85);
///
/// // Add five to the second value.
/// pp.modify(1, 5);
/// assert_eq!(pp.query(0), 10);
/// assert_eq!(pp.query(1), 20);
/// assert_eq!(pp.query(2), 50);
/// assert_eq!(pp.query(3), 90);
///
/// // Multiply every value with 2.
/// pp.map(|v| *v *= 2);
/// assert_eq!(pp.query(0), 20);
/// assert_eq!(pp.query(1), 40);
/// assert_eq!(pp.query(2), 100);
/// assert_eq!(pp.query(3), 180);
///
/// // Divide with two to undo.
/// pp.map(|v| *v /= 2);
/// // Add some more values.
/// pp.extend(vec![0, 10].into_iter());
/// assert_eq!(pp.query(0), 10);
/// assert_eq!(pp.query(1), 20);
/// assert_eq!(pp.query(2), 50);
/// assert_eq!(pp.query(3), 90);
/// assert_eq!(pp.query(4), 90);
/// assert_eq!(pp.query(5), 100);
///
/// // Get the values.
/// assert_eq!(pp.get(0), 10);
/// assert_eq!(pp.get(1), 10);
/// assert_eq!(pp.get(2), 30);
/// assert_eq!(pp.get(3), 40);
/// assert_eq!(pp.get(4), 0);
/// assert_eq!(pp.get(5), 10);
///
/// // Remove the last value
/// pp.truncate(5);
/// assert_eq!(pp.get(0), 10);
/// assert_eq!(pp.get(1), 10);
/// assert_eq!(pp.get(2), 30);
/// assert_eq!(pp.get(3), 40);
/// assert_eq!(pp.get(4), 0);
///
/// // Get back the original values.
/// assert_eq!(pp.into_vec(), vec![10, 10, 30, 40, 0]);
/// ```
///
/// You can also use other operators:
///
/// ```rust
/// use segment_tree::ops::Mul;
/// use segment_tree::PrefixPoint;
///
/// let buf = vec![10, 5, 30, 40];
///
/// let mut pp = PrefixPoint::build(buf, Mul);
///
/// assert_eq!(pp.query(0), 10);
/// assert_eq!(pp.query(1), 50);
/// assert_eq!(pp.query(2), 1500);
/// assert_eq!(pp.query(3), 60000);
/// ```
#[derive(Clone, Hash)]
pub struct PrefixPoint<N, O>
    where O: Commutative<N>
{
    buf: Vec<N>,
    op: O,
}

/// Returns the least significant bit that is one.
#[inline]
fn lsb(i: usize) -> usize {
    i & (1 + !i)
}

/// Could also be done with slice_at_mut, but that's a giant pain
#[inline]
unsafe fn combine_mut<N, O>(buf: &mut Vec<N>, i: usize, j: usize, op: &O)
    where O: Commutative<N>
{
    let ptr1 = &mut buf[i] as *mut N;
    let ptr2 = &buf[j] as *const N;
    op.combine_left_mut(&mut *ptr1, &*ptr2);
}

/// Could also be done with slice_at_mut, but that's a giant pain
#[inline]
unsafe fn invert_mut<N, O>(buf: &mut Vec<N>, i: usize, j: usize, op: &O)
    where O: PartialInvert<N>
{
    let ptr1 = &mut buf[i] as *mut N;
    let ptr2 = &buf[j] as *const N;
    op.invert_mut(&mut *ptr1, &*ptr2);
}

impl<N, O> PrefixPoint<N, O>
    where O: Commutative<N>
{
    /// Creates a `PrefixPoint` containing the given values.
    /// Uses `O(len)` time.
    pub fn build(mut buf: Vec<N>, op: O) -> PrefixPoint<N, O> {
        let len = buf.len();
        for i in 0..len {
            let j = i + lsb(i + 1);
            if j < len {
                unsafe {
                    combine_mut::<N, O>(&mut buf, j, i, &op);
                }
            }
        }
        PrefixPoint { buf: buf, op: op }
    }

    /// Returns the number of values in this tree.
    /// Uses `O(1)` time.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Computes `a[0] * a[1] * ... * a[i]`.  Note that `i` is inclusive.
    /// Uses `O(log(i))` time.
    #[inline]
    pub fn query(&self, mut i: usize) -> N
        where N: Clone
    {
        let mut sum = self.buf[i].clone();
        i -= lsb(1 + i) - 1;
        while i > 0 {
            sum = self.op.combine_left(sum, &self.buf[i - 1]);
            i -= lsb(i);
        }
        sum
    }

    /// Computes `a[0] * a[1] * ... * a[i]`.  Note that `i` is inclusive.
    /// Uses `O(log(i))` time.
    #[inline]
    pub fn query_noclone(&self, mut i: usize) -> MaybeOwned<N> {
        let mut sum = MaybeOwned::Borrowed(&self.buf[i]);
        i -= lsb(1 + i) - 1;
        while i > 0 {
            sum = MaybeOwned::Owned(match sum {
                MaybeOwned::Borrowed(ref v) => self.op.combine(v, &self.buf[i - 1]),
                MaybeOwned::Owned(v) => self.op.combine_left(v, &self.buf[i - 1]),
            });
            i -= lsb(i);
        }
        sum
    }

    /// Combine the value at `i` with `delta`.
    /// Uses `O(log(len))` time.
    #[inline]
    pub fn modify(&mut self, mut i: usize, delta: N) {
        let len = self.len();
        while i < len {
            self.op.combine_left_mut(&mut self.buf[i], &delta);
            i += lsb(i + 1);
        }
    }

    /// Truncates the `PrefixPoint` to the given size.  If `size >= len`, this method does nothing.
    /// Uses `O(1)` time.
    #[inline]
    pub fn truncate(&mut self, size: usize) {
        self.buf.truncate(size);
    }

    /// Calls `shrink_to_fit` on the interval vector.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.buf.shrink_to_fit();
    }

    /// Replace every value in the type with `f(value)`.
    /// This function assumes that `f(a) * f(b) = f(a * b)`.
    /// Applies the function `len` times.
    #[inline]
    pub fn map<F>(&mut self, mut f: F)
        where F: FnMut(&mut N)
    {
        for val in &mut self.buf {
            f(val);
        }
    }
}

impl<N, O> PrefixPoint<N, O>
    where O: PartialInvert<N>
{
    /// Returns the value at `i`.
    /// Uses `O(log(i))` time.
    /// Store your own copy of the array if you want constant time.
    pub fn get(&self, mut i: usize) -> N
        where N: Clone
    {
        let mut sum = self.buf[i].clone();
        let z = 1 + i - lsb(i + 1);
        while i != z {
            sum = self.op.invert(sum, &self.buf[i - 1]);
            i -= lsb(i);
        }
        sum
    }

    /// Compute the underlying array of values.
    /// Uses `O(len)` time.
    pub fn into_vec(self) -> Vec<N> {
        let mut buf = self.buf;
        let len = buf.len();
        for i in (0..len).rev() {
            let j = i + lsb(i + 1);
            if j < len {
                unsafe {
                    invert_mut::<N, O>(&mut buf, j, i, &self.op);
                }
            }
        }
        buf
    }

    pub fn to_vec(&self) -> Vec<N>
        where N: Clone
    {
        let len = self.buf.len();
        let mut buf = self.buf.clone();
        for i in (0..len).rev() {
            let j = i + lsb(i + 1);
            if j < len {
                unsafe {
                    invert_mut::<N, O>(&mut buf, j, i, &self.op);
                }
            }
        }
        buf
    }

}

impl<N, O> PrefixPoint<N, O>
    where O: Invert<N>
{
    /// Change the value at the index to be the specified value.
    /// Uses `O(log(i))` time.
    pub fn set(&mut self, i: usize, value: N)
        where N: Clone
    {
        let current = self.get(i);
        let diff = self.op.invert(value, &current);
        self.modify(i, diff);
    }
}

impl<N, O> Default for PrefixPoint<N, O>
    where O: Commutative<N> + Default
{
    #[inline]
    fn default() -> PrefixPoint<N, O> {
        PrefixPoint {
            buf: Vec::new(),
            op: Default::default(),
        }
    }
}

impl<N, O> Extend<N> for PrefixPoint<N, O>
    where O: Commutative<N>
{
    /// Adds the given values to the `PrefixPoint`, increasing its size.
    /// Uses `O(len)` time.
    fn extend<I: IntoIterator<Item = N>>(&mut self, values: I) {
        let oldlen = self.len();
        self.buf.extend(values);
        let len = self.len();
        for i in 0..len {
            let j = i + lsb(i + 1);
            if oldlen <= j && j < len {
                unsafe {
                    combine_mut::<N, O>(&mut self.buf, j, i, &self.op);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, thread_rng};
    use std::num::Wrapping;
    use ops::Add;

    /// Modifies the given slice such that the n'th element becomes the sum of the first n elements.
    pub fn compute_prefix_sum<N: ::std::ops::Add<Output = N> + Copy>(buf: &mut [N]) {
        let mut iter = buf.iter_mut();
        match iter.next() {
            None => {}
            Some(s) => {
                let mut sum = *s;
                for item in iter {
                    sum = sum + *item;
                    *item = sum;
                }
            }
        }
    }

    #[test]
    fn fenwick_query() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            println!("vec = {:?}", vec);

            let fenwick = PrefixPoint::build(vec.clone(), Add);
            compute_prefix_sum(&mut vec);
            for i in 0..vec.len() {
                assert_eq!(vec[i], fenwick.query(i));
                assert_eq!(&vec[i], fenwick.query_noclone(i).borrow());
            }
        }
    }
    #[test]
    fn fenwick_map() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            println!("vec = {:?}", vec);

            let mut fenwick = PrefixPoint::build(vec.clone(), Add);
            assert_eq!(fenwick.clone().into_vec(), vec);
            assert_eq!(fenwick.clone().to_vec(), vec);
            compute_prefix_sum(&mut vec);
            fenwick.map(|n| *n = Wrapping(12) * *n);
            for i in 0..vec.len() {
                assert_eq!(vec[i] * Wrapping(12), fenwick.query(i));
            }
        }
    }
    #[test]
    fn fenwick_modify() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let diff: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            println!("vec = {:?}", vec);
            println!("diff = {:?}", diff);

            let mut fenwick = PrefixPoint::build(vec.clone(), Add);
            for i in 0..diff.len() {
                let mut ps: Vec<Wrapping<i32>> = vec.clone();
                compute_prefix_sum(&mut ps);
                assert_eq!(fenwick.clone().into_vec(), vec);
                assert_eq!(fenwick.clone().to_vec(), vec);
                for j in 0..vec.len() {
                    assert_eq!(ps[j], fenwick.query(j));
                    assert_eq!(vec[j], fenwick.get(j));
                }
                vec[i] += diff[i];
                fenwick.modify(i, diff[i]);
            }
        }
    }

    #[test]
    fn fenwick_set() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let diff: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            println!("vec = {:?}", vec);
            println!("diff = {:?}", diff);

            let mut fenwick = PrefixPoint::build(vec.clone(), Add);
            for i in 0..diff.len() {
                let mut ps: Vec<Wrapping<i32>> = vec.clone();
                compute_prefix_sum(&mut ps);
                assert_eq!(fenwick.clone().into_vec(), vec);
                assert_eq!(fenwick.clone().to_vec(), vec);
                for j in 0..vec.len() {
                    assert_eq!(ps[j], fenwick.query(j));
                    assert_eq!(vec[j], fenwick.get(j));
                }
                vec[i] = diff[i];
                fenwick.set(i, diff[i]);
            }
        }
    }

    #[test]
    fn fenwick_extend() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let vec: Vec<Wrapping<i32>> =
                rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let mut sum = vec.clone();
            compute_prefix_sum(&mut sum);
            for i in 0..sum.len() {
                let mut fenwick = PrefixPoint::build(vec.iter().take(i / 2).map(|&i| i).collect(),
                                                     Add);
                fenwick.extend(vec.iter().skip(i / 2).take(i - i / 2).map(|&i| i));
                for j in 0..i {
                    assert_eq!(sum[j], fenwick.query(j));
                }
            }
        }
    }
}
