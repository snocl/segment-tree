use ops::{CommutativeOperation, Inverse};
use std::marker::PhantomData;

/// This data structure allows prefix queries and single element modification.
///
/// This tree allocates `n * sizeof(N)` bytes of memory, and can be resized.
///
/// This data structure is implemented using a fenwick tree, which is also known as a binary
/// indexed tree.
pub struct PrefixPoint<N, O> where O: CommutativeOperation<N> {
    buf: Vec<N>,
    op: PhantomData<O>
}

/// Returns the least significant bit which is one.
#[inline(always)]
fn lsb(i: usize) -> usize {
    i & (1 + !i)
}

/// Could also be done with slice_at_mut, but that's a giant pain
#[inline(always)]
unsafe fn combine_mut<N, O: CommutativeOperation<N>>(buf: &mut Vec<N>, i: usize, j: usize) {
    let ptr1 = &mut buf[i] as *mut N;
    let ptr2 = &buf[j] as *const N;
    O::combine_mut(&mut *ptr1, &*ptr2);
}
/// Could also be done with slice_at_mut, but that's a giant pain
#[inline(always)]
unsafe fn uncombine_mut<N, O: Inverse<N>>(buf: &mut Vec<N>, i: usize, j: usize) {
    let ptr1 = &mut buf[i] as *mut N;
    let ptr2 = &buf[j] as *const N;
    O::uncombine(&mut *ptr1, &*ptr2);
}

impl<N, O: CommutativeOperation<N>> PrefixPoint<N, O> {
    /// Creates a `PrefixPoint` containing the given values.
    /// Uses `O(len)` time.
    pub fn build(mut buf: Vec<N>) -> PrefixPoint<N, O> {
        let len = buf.len();
        for i in 0..len {
            let j = i + lsb(i+1);
            if j < len {
                unsafe {
                    combine_mut::<N, O>(&mut buf, j, i);
                }
            }
        }
        PrefixPoint { buf: buf, op: PhantomData }
    }
    /// Returns the number of values in this tree.
    /// Uses `O(1)` time.
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    /// Computes `a[0] * a[1] * ... * a[i]`.  Note that `i` is inclusive.
    /// Uses `O(log(i))` time.
    #[inline]
    pub fn query(&self, mut i: usize) -> N where N: Clone {
        let mut sum = self.buf[i].clone();
        i -= lsb(1+i) - 1;
        while i > 0 {
            sum = O::combine_left(sum, &self.buf[i-1]);
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
            O::combine_mut(&mut self.buf[i], &delta);
            i += lsb(i+1);
        }
    }
    /// Truncates the `PrefixPoint` to the given size.  If `size >= len`, this method does nothing.
    /// Uses `O(1)` time.
    #[inline(always)]
    pub fn truncate(&mut self, size: usize) {
        self.buf.truncate(size);
    }
    /// Replace every value in the type with `f(value)`.
    /// This function assumes that `f(a) * f(b) = f(a * b)`.
    /// Applies then function `len` times.
    #[inline]
    pub fn map<F: FnMut(&mut N)>(&mut self, mut f: F) {
        for val in &mut self.buf {
            f(val);
        }
    }
    /// Adds the given values to the `PrefixPoint`, increasing its size.
    /// Uses `O(len)` time.
    #[inline]
    pub fn append(&mut self, mut values: Vec<N>) {
        self.extend(values.drain(..))
    }
}
impl<N, O: CommutativeOperation<N>> Extend<N> for PrefixPoint<N, O> {
    /// Adds the given values to the `PrefixPoint`, increasing its size.
    /// Uses `O(len)` time.
    fn extend<I: IntoIterator<Item=N>>(&mut self, values: I) {
        let oldlen = self.len();
        self.buf.extend(values);
        let len = self.len();
        for i in 0..len {
            let j = i + lsb(i+1);
            if oldlen <= j && j < len {
                unsafe {
                    combine_mut::<N, O>(&mut self.buf, j, i);
                }
            }
        }
    }
}
impl<N, O: CommutativeOperation<N> + Inverse<N>> PrefixPoint<N, O> {
    /// Returns the value at `i`.
    /// Uses `O(log(i))` time.
    /// Store your own copy of the array if you want constant time.
    pub fn get(&self, mut i: usize) -> N where N: Clone {
        let mut sum = self.buf[i].clone();
        let z = 1 + i - lsb(i+1);
        while i != z {
            O::uncombine(&mut sum, &self.buf[i-1]);
            i -= lsb(i);
        }
        sum
    }
    /// Compute the underlying array of values.
    /// Uses `O(len)` time.
    pub fn unwrap(self) -> Vec<N> {
        let mut buf = self.buf;
        let len = buf.len();
        for i in (0..len).rev() {
            let j = i + lsb(i+1);
            if j < len {
                unsafe {
                    uncombine_mut::<N, O>(&mut buf, j, i);
                }
            }
        }
        buf
    }
}

impl<N: Clone, O: CommutativeOperation<N>> Clone for PrefixPoint<N, O> {
    fn clone(&self) -> PrefixPoint<N, O> {
        PrefixPoint {
            buf: self.buf.clone(), op: PhantomData
        }
    }
}

#[cfg(test)]
mod tests {

    /// Modifies the given slice such that the n'th element becomes the sum of the first n elements.
    pub fn compute_prefix_sum<N: ::std::ops::Add<Output=N> + Copy>(buf: &mut[N]) {
        let mut iter = buf.iter_mut();
        match iter.next() {
            None => {},
            Some(s) => {
                let mut sum = *s;
                for item in iter {
                    sum = sum + *item;
                    *item = sum;
                }
            }
        }
    }

    use super::*;
    use rand::{Rng, thread_rng};
    use std::num::Wrapping;
    use ops::Add;

    #[test]
    fn fenwick_query() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> = rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let fenwick: PrefixPoint<_, Add> = PrefixPoint::build(vec.clone());
            compute_prefix_sum(&mut vec);
            for i in 0..vec.len() {
                assert_eq!(vec[i], fenwick.query(i));
            }
        }
    }
    #[test]
    fn fenwick_map() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> = rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let mut fenwick: PrefixPoint<_, Add> = PrefixPoint::build(vec.clone());
            assert_eq!(fenwick.clone().unwrap(), vec);
            compute_prefix_sum(&mut vec);
            fenwick.map(|n| *n = Wrapping(12) * *n);
            for i in 0..vec.len() {
                assert_eq!(vec[i]*Wrapping(12), fenwick.query(i));
            }
        }
    }
    #[test]
    fn fenwick_modify() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let mut vec: Vec<Wrapping<i32>> = rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let diff: Vec<Wrapping<i32>> = rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let mut fenwick: PrefixPoint<_, Add> = PrefixPoint::build(vec.clone());
            for i in 0..diff.len() {
                let mut ps: Vec<Wrapping<i32>> = vec.clone();
                compute_prefix_sum(&mut ps);
                assert_eq!(fenwick.clone().unwrap(), vec);
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
    fn fenwick_extend() {
        let mut rng = thread_rng();
        for n in 0..130 {
            let vec: Vec<Wrapping<i32>> = rng.gen_iter::<i32>().take(n).map(|i| Wrapping(i)).collect();
            let mut sum = vec.clone();
            compute_prefix_sum(&mut sum);
            for i in 0..sum.len() {
                let mut fenwick: PrefixPoint<_, Add> = PrefixPoint::build(vec.iter().take(i/2).map(|&i| i).collect());
                fenwick.extend(vec.iter().skip(i/2).take(i - i/2).map(|&i| i));
                for j in 0..i {
                    assert_eq!(sum[j], fenwick.query(j));
                }
            }
        }
    }
}