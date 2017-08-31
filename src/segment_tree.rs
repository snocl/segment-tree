use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::mem;

use maybe_owned::MaybeOwned::{self, Owned, Borrowed};
use ops::{Operation, Commutative, Identity};

/// This data structure allows range queries and single element modification.
///
/// This tree allocates `2n * sizeof(N)` bytes of memory.
///
/// This tree is implemented using a segment tree.  A segment tree is a binary tree where each node
/// contains the combination of the children under the operation.
///
/// # Examples
///
/// This data structure allows solving the [range minimum query][1] problem in logaritmic time.
///
/// ```rust
/// use segment_tree::SegmentPoint;
/// use segment_tree::ops::Min;
///
/// // make a segment point
/// let mut tree = SegmentPoint::build(
///     vec![10, 5, 6, 4, 12, 8, 9, 3, 2, 1, 5], Min
/// ); //     0  1  2  3   4  5  6  7  8  9 10  - indices
///
/// // find the minimum value in a few intervals
/// assert_eq!(Some(5), tree.query(0, 2));
/// assert_eq!(Some(3), tree.query(4, 8));
/// assert_eq!(Some(1), tree.query(3, 11));
/// assert_eq!(Some(1), tree.query(0, 11));
///
/// // query returns None if given an invalid interval
/// assert_eq!(None, tree.query(4, 2));
///
/// // if we want to avoid cloning, we can use query_noclone:
/// use segment_tree::maybe_owned::MaybeOwned;
/// assert_eq!(Some(MaybeOwned::Owned(2)), tree.query_noclone(4, 9));
///
/// // since minimum is commutative, we can use query_commut
/// assert_eq!(1, tree.query_commut(3, 11));
///
/// // let's update a few values
/// tree.modify(1, 3);
/// assert_eq!(3, tree.query_commut(0, 2));
/// assert_eq!(3, tree.query_commut(5, 8));
///
/// tree.modify(3, 0);
/// assert_eq!(0, tree.query_commut(2, 8));
///
/// // we can view the values currently stored at any time
/// assert_eq!(&[10, 3, 6, 0, 12, 8, 9, 3, 2, 1, 5], tree.view());
/// ```
///
/// We can also use a `SegmentPoint` to find the sum of any interval, by changing the operator to
/// [`Add`].
///
/// ```rust
/// use segment_tree::SegmentPoint;
/// use segment_tree::ops::Add;
///
/// // make a segment point
/// let mut tree = SegmentPoint::build(
///    vec![10, 5, 6, 4, 12, 8, 9, 3, 2, 1, 5], Add
/// ); //     0  1  2  3   4  5  6  7  8  9 10  - indices
///
/// assert_eq!(12 + 8 + 9 + 3, tree.query_commut(4, 8));
/// assert_eq!(5 + 6, tree.query_commut(1, 3));
///
/// // query_commut returns the identity if given an invalid interval
/// assert_eq!(0, tree.query_commut(3, 1));
///
/// // we can still modify values in the tree
/// tree.modify(2, 4);
/// assert_eq!(5 + 4, tree.query_commut(1, 3));
/// assert_eq!(12 + 8 + 9 + 3, tree.query_commut(4, 8));
///
/// assert_eq!(&[10, 5, 4, 4, 12, 8, 9, 3, 2, 1, 5], tree.view());
/// ```
///
/// [1]: https://en.wikipedia.org/wiki/Range_minimum_query
/// [`Add`]: ops/struct.Add.html
pub struct SegmentPoint<N, O>
    where O: Operation<N>
{
    buf: Vec<N>,
    n: usize,
    op: O,
}

impl<N, O: Operation<N>> SegmentPoint<N, O> {
    /// Builds a tree using the given buffer.  If the given buffer is less than half full, this
    /// function allocates.  This function clones every value in the input array.
    /// Uses `O(len)` time.
    ///
    /// See also the function [`build_noalloc`].
    ///
    /// [`build_noalloc`]: struct.SegmentPoint.html#method.build_noalloc
    pub fn build(mut buf: Vec<N>, op: O) -> SegmentPoint<N, O>
        where N: Clone
    {
        let n = buf.len();
        buf.reserve_exact(n);
        for i in 0..n {
            debug_assert!(i < buf.len());
            let clone = unsafe { buf.get_unchecked(i).clone() }; // i < n < buf.len()
            buf.push(clone);
        }
        SegmentPoint::build_noalloc(buf, op)
    }

    /// Computes `a[l] * a[l+1] * ... * a[r-1]`.
    ///
    /// If `l >= r`, this method returns `None`.  This method clones at most twice and runs in
    /// `O(log(len))` time.
    ///
    /// See [`query_commut`] or [`query_noclone`] for a version that doesn't clone.
    ///
    /// [`query_noclone`]: struct.SegmentPoint.html#method.query_noclone
    /// [`query_commut`]: struct.SegmentPoint.html#method.query_commut
    pub fn query(&self, mut l: usize, mut r: usize) -> Option<N>
        where N: Clone
    {
        let mut resl = None;
        let mut resr = None;
        l += self.n;
        r += self.n;
        while l < r {
            if l & 1 == 1 {
                resl = Some(match resl {
                    None => self.buf[l].clone(),
                    Some(mut left) => {
                        self.op.combine_mut(&mut left, &self.buf[l]);
                        left
                    }
                });
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                resr = Some(match resr {
                    None => self.buf[r].clone(),
                    Some(mut right) => {
                        self.op.combine_mut_right(&self.buf[r], &mut right);
                        right
                    }
                });
            }
            l >>= 1;
            r >>= 1;
        }

        match (resl, resr) {
            (None, result) | (result, None) => result,
            (Some(mut left), Some(right)) => {
                self.op.combine_mut(&mut left, &right);
                Some(left)
            }
        }
    }

    /// Computes `a[l] * a[l+1] * ... * a[r-1]`.
    ///
    /// If `l >= r`, this method returns `None`.
    /// Uses `O(log(len))` time.
    ///
    /// See also [`query`] and [`query_commut`].
    ///
    /// [`query_commut`]: struct.SegmentPoint.html#method.query_commut
    /// [`query`]: struct.SegmentPoint.html#method.query
    pub fn query_noclone<'a>(&'a self, mut l: usize, mut r: usize) -> Option<MaybeOwned<'a, N>> {
        let mut resl = None;
        let mut resr = None;
        l += self.n;
        r += self.n;
        while l < r {
            if l & 1 == 1 {
                let right = &self.buf[l];
                resl = Some(match resl {
                    None => Borrowed(right),
                    Some(Borrowed(left)) => Owned(self.op.combine(left, right)),
                    Some(Owned(mut left)) => {
                        self.op.combine_mut(&mut left, right);
                        Owned(left)
                    }
                });
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                let left = &self.buf[r];
                resr = Some(match resr {
                    None => Borrowed(left),
                    Some(Borrowed(ref right)) => Owned(self.op.combine(left, right)),
                    Some(Owned(mut right)) => {
                        self.op.combine_mut_right(left, &mut right);
                        Owned(right)
                    }
                });
            }
            l >>= 1;
            r >>= 1;
        }

        match resl {
            None => resr,
            Some(Owned(mut left)) => {
                if let Some(ref right) = resr {
                    self.op.combine_mut(&mut left, right.as_ref());
                }
                Some(Owned(left))
            }
            Some(Borrowed(left)) => {
                let right = match resr {
                    Some(right) => right,
                    None => return Some(Borrowed(left)),
                };

                Some(Owned(match right {
                    Borrowed(right) => self.op.combine(left, right),
                    Owned(mut right) => {
                        self.op.combine_mut_right(left, &mut right);
                        right
                    }
                }))
            }
        }
    }

    /// Set the value at the specified index and return the old value.
    /// Uses `O(log(len))` time.
    pub fn modify(&mut self, mut p: usize, value: N) -> N {
        // TODO: Is it possible to take value by ref?

        p += self.n;
        let res = mem::replace(&mut self.buf[p], value);

        while p > 1 {
            let lp = p & !1;
            let rp = p | 1;
            p >>= 1;

            self.buf[p] = self.op.combine(&self.buf[lp], &self.buf[rp]);
        }
        res
    }

    /// Combine the value at `p` with `delta`, such that `delta` is the left argument.
    /// Uses `O(log(len))` time.
    pub fn compose_left(&mut self, mut p: usize, delta: &N) {
        p += self.n;
        self.op.combine_mut_right(delta, &mut self.buf[p]);

        while p > 1 {
            let lp = p & !1;
            let rp = p | 1;
            p >>= 1;

            self.buf[p] = self.op.combine(&self.buf[lp], &self.buf[rp]);
        }
    }

    /// Combine the value at `p` with `delta`, such that `delta` is the right argument.
    /// Uses `O(log(len))` time.
    pub fn compose_right(&mut self, mut p: usize, delta: &N) {
        p += self.n;
        self.op.combine_mut(&mut self.buf[p], delta);

        while p > 1 {
            let lp = p & !1;
            let rp = p | 1;
            p >>= 1;

            self.buf[p] = self.op.combine(&self.buf[lp], &self.buf[rp]);
        }
    }

    /// View the values in this segment tree using a slice.  Uses `O(1)` time.
    #[inline]
    pub fn view(&self) -> &[N] {
        &self.buf[self.n..]
    }

    /// The number of elements stored in this segment tree.  Uses `O(1)` time.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Builds a tree using the given buffer.  The buffer must be even in size.
    /// The first `n` values have no effect on the resulting tree,
    /// and the remaining `n` values contains the array to build the tree on.
    /// Uses `O(len)` time.
    ///
    /// This function panics if the size of the buffer is odd.
    ///
    ///# Example
    ///
    ///```rust
    ///use segment_tree::SegmentPoint;
    ///use segment_tree::ops::Min;
    ///
    /// // make a segment point using the other build method
    ///let mut tree = SegmentPoint::build(
    ///    vec![1, 2, 3, 4], Min
    ///);
    /// // make a segment point using the build_noalloc method:
    ///let mut tree1 = SegmentPoint::build_noalloc(
    ///    vec![3282, 210, 0, 245, 1, 2, 3, 4], Min  // the first half of the values are ignored
    ///);
    ///assert_eq!(tree, tree1);
    ///
    ///let mut tree2 = SegmentPoint::build_noalloc(
    ///    vec![0, 0, 0, 0, 1, 2, 3, 4], Min  // we can also try some other first few values
    ///);
    ///assert_eq!(tree1, tree2);
    ///assert_eq!(tree, tree2);
    ///```
    pub fn build_noalloc(mut buf: Vec<N>, op: O) -> SegmentPoint<N, O> {
        let len = buf.len();
        let n = len >> 1;
        if len & 1 == 1 {
            panic!("SegmentPoint::build_noalloc: odd size");
        }
        for i in (1..n).rev() {
            let res = op.combine(&buf[i << 1], &buf[i << 1 | 1]);
            buf[i] = res;
        }
        SegmentPoint {
            buf: buf,
            op: op,
            n: n,
        }
    }
}

impl<N, O: Commutative<N>> SegmentPoint<N, O> {
    /// Combine the value at `p` with `delta`.
    /// Uses `O(log(len))` time.
    #[inline]
    pub fn compose(&mut self, p: usize, delta: &N) {
        self.compose_right(p, delta);
    }
}

impl<N, O: Commutative<N> + Identity<N>> SegmentPoint<N, O> {
    /// Computes `a[l] * a[l+1] * ... * a[r-1]`.
    /// Uses `O(log(len))` time.
    ///
    /// If `l >= r`, this method returns the identity.
    ///
    /// See [`query`] or [`query_noclone`] for a version that works with
    /// non-[commutative operations][1].
    ///
    /// [`query`]: struct.SegmentPoint.html#method.query
    /// [`query_noclone`]: struct.SegmentPoint.html#method.query_noclone
    /// [1]: ops/trait.Commutative.html
    pub fn query_commut(&self, mut l: usize, mut r: usize) -> N {
        let mut res = self.op.identity();
        l += self.n;
        r += self.n;
        while l < r {
            if l & 1 == 1 {
                self.op.combine_mut(&mut res, &self.buf[l]);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                self.op.combine_mut(&mut res, &self.buf[r]);
            }
            l >>= 1;
            r >>= 1;
        }
        res
    }
}

impl<N: Clone, O: Operation<N> + Clone> Clone for SegmentPoint<N, O> {
    #[inline]
    fn clone(&self) -> SegmentPoint<N, O> {
        SegmentPoint {
            buf: self.buf.clone(),
            n: self.n,
            op: self.op.clone(),
        }
    }
}

impl<N: Debug, O: Operation<N> + Debug> Debug for SegmentPoint<N, O> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SegmentPoint")
            .field("view", &self.view())
            .field("op", &self.op)
            .finish()
    }
}

impl<N: PartialEq, O: Operation<N> + PartialEq> PartialEq for SegmentPoint<N, O> {
    #[inline]
    fn eq(&self, other: &SegmentPoint<N, O>) -> bool {
        self.op.eq(&other.op) && self.view().eq(other.view())
    }

    #[inline]
    fn ne(&self, other: &SegmentPoint<N, O>) -> bool {
        self.op.ne(&other.op) || self.view().ne(other.view())
    }
}

impl<N: Eq, O: Operation<N> + Eq> Eq for SegmentPoint<N, O> {}

impl<N, O: Operation<N> + Default> Default for SegmentPoint<N, O> {
    #[inline]
    fn default() -> SegmentPoint<N, O> {
        SegmentPoint {
            buf: Vec::new(),
            n: 0,
            op: Default::default(),
        }
    }
}

impl<N: Hash, O: Operation<N> + Hash> Hash for SegmentPoint<N, O> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.view().hash(state);
        self.op.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use rand::{Rand, Rng, thread_rng};

    use maybe_owned::MaybeOwned;
    use ops::*;
    use super::*;

    // TODO: Remove these particularly unhelpful uses of random data in tests.

    /// Not commutative! Not useful in practice since the root always contains the concatenation of
    /// every string.
    #[derive(PartialEq, Eq, Clone, Debug)]
    struct StrType {
        value: String,
    }

    impl StrType {
        fn cat(list: &[StrType]) -> StrType {
            let mut res = String::new();
            for v in list {
                res.push_str(v.value.as_str());
            }
            StrType { value: res }
        }
        fn sub(&self, i: usize, j: usize) -> StrType {
            StrType { value: String::from(&self.value[4 * i..4 * j]) }
        }
    }

    impl Rand for StrType {
        fn rand<R: Rng>(rng: &mut R) -> Self {
            StrType { value: rng.gen_ascii_chars().take(4).collect() }
        }
    }

    impl Operation<StrType> for Add {
        fn combine(&self, a: &StrType, b: &StrType) -> StrType {
            let mut a = a.clone();
            self.combine_mut(&mut a, b);
            a
        }

        fn combine_mut(&self, a: &mut StrType, b: &StrType) {
            a.value.push_str(b.value.as_str());
        }
    }

    #[test]
    fn segment_tree_build() {
        let mut rng = thread_rng();
        let vals: Vec<_> = rng.gen_iter::<i32>().map(|i| Wrapping(i)).take(130).collect();

        println!("vals = {:?}", vals);

        for i in 0..vals.len() {
            let buf: Vec<_> = vals[0..i].iter().cloned().collect();
            println!("{:?}", buf);
            let mut buf2 = vec![];
            let n = buf.len();
            buf2.resize(2 * n, Wrapping(0));
            for i in 0..n {
                buf2[n + i] = buf[i];
            }
            let tree1 = SegmentPoint::build(buf, Add);
            let tree2 = SegmentPoint::build_noalloc(buf2, Add);
            let mut buf = tree1.buf;
            let mut buf2 = tree2.buf;
            if i > 0 {
                buf[0] = Wrapping(0);
                buf2[0] = Wrapping(0);
            }
            println!("build");
            println!("{:?}", buf);
            println!("build_noalloc");
            println!("{:?}", buf2);
            assert_eq!(buf, buf2);
            assert_eq!(2 * n, buf.len());
        }
    }

    #[test]
    fn segment_tree_query() {
        let mut rng = thread_rng();
        let vals: Vec<StrType> = rng.gen_iter().take(130).collect();

        println!("vals = {:?}", vals);

        for i in 0..vals.len() {
            let buf: Vec<_> = vals[0..i].iter().cloned().collect();
            let tree = SegmentPoint::build(buf.clone(), Add);
            let sum = StrType::cat(&buf);
            let n = buf.len();
            println!("n: {} tree.buf.len: {}", n, tree.buf.len());
            for i in 0..n {
                assert_eq!(None, tree.query(i, i));
                for j in i + 1..n + 1 {
                    println!("i: {}, j: {}", i, j);
                    assert_eq!(Some(sum.sub(i, j)), tree.query(i, j));
                    assert_eq!(Some(MaybeOwned::Owned(sum.sub(i, j))),
                               tree.query_noclone(i, j));
                }
            }
        }
    }

    #[test]
    fn segment_tree_query_commut() {
        let mut rng = thread_rng();
        let vals: Vec<Wrapping<i32>> = rng.gen_iter().map(|i| Wrapping(i)).take(130).collect();

        println!("vals = {:?}", vals);

        for i in 0..vals.len() {
            let mut buf: Vec<_> = vals[0..i].iter().cloned().collect();
            let tree = SegmentPoint::build(buf.clone(), Add);
            assert_eq!(&buf[..], tree.view());
            for i in 1..buf.len() {
                buf[i] += buf[i - 1];
            } // compute the prefix sum
            let n = buf.len();
            println!("n: {} tree.buf.len: {}", n, tree.buf.len());
            for i in 0..n {
                assert_eq!(None, tree.query(i, i));
                for j in i + 1..n + 1 {
                    println!("i: {}, j: {}", i, j);
                    if i == 0 {
                        assert_eq!(buf[j - 1], tree.query_commut(i, j));
                    } else {
                        assert_eq!(buf[j - 1] - buf[i - 1], tree.query_commut(i, j));
                    }
                }
            }
        }
    }

    #[test]
    fn segment_tree_modify() {
        let mut rng = thread_rng();
        let vals1: Vec<Wrapping<i32>> =
            rng.gen_iter::<i32>().map(|i| Wrapping(i)).take(130).collect();
        let vals2: Vec<Wrapping<i32>> =
            rng.gen_iter::<i32>().map(|i| Wrapping(i)).take(130).collect();

        println!("vals1 = {:?}", vals1);
        println!("vals2 = {:?}", vals2);

        for i in 0..vals1.len() {
            let mut order: Vec<_> = (0..i).collect();
            rng.shuffle(&mut order[..]);

            println!("order = {:?}", order);

            let mut buf: Vec<_> = vals1[0..i].iter().cloned().collect();
            let mut tree = SegmentPoint::build(buf.clone(), Add);
            for next in order {
                tree.modify(next, vals2[next]);
                buf[next] = vals2[next];
                let tree2 = SegmentPoint::build(buf.clone(), Add);
                assert_eq!(tree.buf[1..], tree2.buf[1..]);
            }
        }
    }
}
