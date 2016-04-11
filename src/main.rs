/* main.rs
 *
 * Copyright (C) 2016 Zheguang Zhao
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 */
use std::cmp;
use std::f32;
use std::collections::HashMap;

extern crate rand;
use rand::Rng;

type TupleId = usize;
type Datum = u32;

#[derive(PartialEq, Eq, Debug)]
struct Rtree {
    root: Node,
    table: Table,
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct Table {
    cols: Vec<Column>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct Column {
    // Should suppot lookup a tuple value by tid 
    // the physical ordering of the values in the column should be oblivious to the caller
    xs: Vec<Datum>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct Point {
    xs: Vec<Datum>,
}

#[derive(PartialEq, Eq, Debug)]
struct Rect {
    min: Point,
    max: Point,
    count: usize,
}

#[derive(PartialEq, Eq, Debug)]
enum Node {
    Leaf {
        tids: Vec<TupleId>,
    },
    Inner {
        mbrs: Vec<Rect>,
        vs: Vec<Node>,
    },
}

impl Table {
    fn new() -> Table {
        Table { cols: vec![] }
    }

    fn len(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.cols[0].len()
        }
    }

    fn is_empty(&self) -> bool {
        self.cols.is_empty() || self.cols[0].is_empty()
    }

    fn width(&self) -> usize {
        self.cols.len()
    }

    fn tids(&self) -> Vec<TupleId> {
        (0..self.len()).collect::<Vec<TupleId>>()
    }

    /**
     * A priority list of orders that physical split point considers.
     */
    fn orders(&self) -> Vec<usize> {
        (0..self.width()).collect::<Vec<usize>>()
    }

    fn add(&mut self, col: Column) {
        self.cols.push(col);
    }
}

impl Column {
    fn val(&self, tid: TupleId) -> Datum {
        self.xs[tid]
    }

    fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    fn len(&self) -> usize {
        self.xs.len()
    }
}

impl Point {
    fn of_tuple(tid: usize, table: &Table) -> Point {
        Point {
            xs: table.cols.iter().map(|c| c.val(tid)).collect::<Vec<Datum>>()
        }
    }
}

impl Rect {
    fn mbr_of(tids: &[TupleId], table: &Table) -> Rect {
        assert!(!tids.is_empty());
        let mut min = Point::of_tuple(tids[0], table);
        let mut max = min.clone();
        for &tid in tids.iter().skip(1) {
            let p = Point::of_tuple(tid, table);
            for (i, &x) in p.xs.iter().enumerate() {
                min.xs[i] = cmp::min(min.xs[i], x);
                max.xs[i] = cmp::max(max.xs[i], x);
            }
        }
        Rect { min: min, max: max, count: tids.len() }
    }

    fn of_p(p: &Point) -> Rect {
        Rect { min: p.clone(), max: p.clone(), count: 1 }
    }

    fn overlap(&self, other: &Rect) -> bool {
        !(
            self.min.xs.iter().zip(other.max.xs.iter()).any(|(&min1, &max2)| min1 > max2)
            || other.min.xs.iter().zip(self.max.xs.iter()).any(|(&min2, &max1)| min2 > max1)
         )
    }

    fn area(&self) -> f32 {
        self.max.xs.iter().zip(self.min.xs.iter()).fold(0f32, |ln_prod, (&a, &b)| ln_prod + (a as f32 - b as f32).ln())
    }

    fn diag_vector(&self) -> Point {
        Point {
            xs: self.max.xs.iter().zip(self.min.xs.iter()).map(|(h, l)| h - l).collect::<Vec<Datum>>()
        }
    }
}

impl Rtree {
    fn new() -> Rtree {
        Rtree { root: Node::Leaf { tids: vec![] }, table: Table::new() }
    }

    fn rtree_on_table(table: Table, phys_fanout: usize) -> Rtree {
        assert!(!table.is_empty());
        let mut res = Rtree::new();
        res.table = table;
        res.phys_bulkload(phys_fanout);
        res
    }

    fn box_search(&self, bb: &Rect) -> Vec<TupleId> {
        Rtree::_box_search(&self.root, bb, &self.table)
    }

    /**
     * Finds a selection of candidate grid elements which may be contained in a given bounding box (or
     * rectangle) BB. The algorithm is called on an initial tree node r, which usually initially the
     * root node.
     */
    fn _box_search(r: &Node, bb: &Rect, table: &Table) -> Vec<TupleId> {
        let mut res = vec![];
        match *r {
            Node::Leaf { ref tids, } => {
                for &tid in tids {
                    let p = Point::of_tuple(tid, table);
                    if Rect::of_p(&p).overlap(bb) {
                        res.push(tid);
                    }
                }
            },
            Node::Inner { ref mbrs, ref vs } => {
                for (v, mbr) in vs.iter().zip(mbrs.iter()) {
                    if mbr.overlap(bb) {
                        res.extend(Rtree::_box_search(v, bb, table).iter().cloned());
                    }
                }
            }
        };
        res
    }

    /**
     * Loads an rtree to root.
     */
    fn phys_bulkload(&mut self, phys_fanout: usize) {
        assert!(!self.table.is_empty());
        let h = cmp::max(0, ((self.table.len() as f32).ln() / (phys_fanout as f32).ln()).ceil() as u32 - 1) as usize;
        self.root = Rtree::phys_bulkload_chunk(self.table.tids(), &self.table, phys_fanout, &self.table.orders(), h);
    }

    /**
     * This algorithm builds an R-tree with tree depth, according to a set of imposed orderings on MBRs.
     */
    fn phys_bulkload_chunk(tids: Vec<TupleId>, table: &Table, phys_fanout: usize, orders: &[usize], h: usize) -> Node {
        assert!(!tids.is_empty());
        if h == 0 {
            Node::Leaf { tids: tids }
        } else {
            let max_ps_per_subtree = phys_fanout.pow(h as u32) as usize;
            let parts = Rtree::phys_partition(tids, table, max_ps_per_subtree, orders);
            assert!(parts.len() <= phys_fanout);
            let mut mbrs = Vec::new();
            let mut vs = Vec::new();
            for part in parts {
                mbrs.push(Rect::mbr_of(&part, table));
                vs.push(Rtree::phys_bulkload_chunk(part, table, phys_fanout, orders, h - 1));
            }
            Node::Inner { mbrs: mbrs, vs: vs }
        }
    }

    /**
     * This method partitions an input set into k ≤ M subsets using the cost function 'fc' and the
     * orderings contained in 'orders'.
     */
    fn phys_partition(tids: Vec<TupleId>, table: &Table, max_ps_per_subtree: usize, orders: &[usize]) -> Vec<Vec<TupleId>> {
        assert!(!tids.is_empty());
        if tids.len() <= max_ps_per_subtree {
            vec![tids]
        } else {
            let mut parts: Vec<Vec<TupleId>> = Vec::new();
            let (lpart, hpart) = Rtree::phys_best_binary_split(tids, table, max_ps_per_subtree, orders);
            let lparts = Rtree::phys_partition(lpart, table, max_ps_per_subtree, orders);
            let hparts = Rtree::phys_partition(hpart, table, max_ps_per_subtree, orders);
            parts.extend(lparts);
            parts.extend(hparts);
            parts
        }
    }

    /**
     * This method splits an input set into two sets. This is done by cutting input set in partitions
     * after first having sorted the set for an ordering. By comparing the cost function applied to
     * each pair of partition and selecting the pair resulting in the lowest value, we obtain a best
     * binary split. The best binary split resulting from all different orderings is then returned.
     */
    fn phys_best_binary_split(tids_: Vec<TupleId>, table: &Table, max_ps_per_subtree: usize, orders: &[usize]) -> (Vec<TupleId>, Vec<TupleId>) {
        assert!(!tids_.is_empty());
        let mut tids = tids_;
        let n = tids.len();
        let num_subtrees = (((n as f32) / (max_ps_per_subtree as f32)).ceil() as u32 - 1) as usize;

        let mut cost = f32::MAX;
        let mut split = 0 as usize;
        let mut new_best = false;
        let mut lpart = Vec::default();
        let mut hpart = Vec::default();

        for &s in orders {
            tids.sort_by(|&a, &b| table.cols[s].val(a).cmp(&table.cols[s].val(b)));

            for i in 1..(num_subtrees+1) {
                let b1 = Rect::mbr_of(&tids[0..(max_ps_per_subtree*i)], table);
                let b2 = Rect::mbr_of(&tids[(max_ps_per_subtree*i)..n], table);
                let t = Rtree::fc(&b1, &b2);
                if t < cost {
                    new_best = true;
                    cost = t;
                    split = i;
                }
            }
            if new_best {
                new_best = false;
                let (lpart_, hpart_) = tids.split_at(max_ps_per_subtree * split);
                lpart = lpart_.to_vec();
                hpart = hpart_.to_vec();
            }
        }

        (lpart, hpart)
    }

    fn fc(a: &Rect, b: &Rect) -> f32 {
        (a.area() + b.area()) / 2f32
    }

    fn stats(&self) -> HashMap<String,usize> {
        let mut res = HashMap::new();
        res.insert("table width".to_string(), self.table.width());
        res.insert("table length".to_string(), self.table.len());

        let levels = self.levels();

        res.insert("tree height".to_string(), levels.len());
        for i in 0..levels.len() {
            res.insert("level ".to_string() + &i.to_string(), levels.get(&i).unwrap().len());
        }

        res
    }

    fn levels<'a>(&'a self) -> HashMap<usize,Vec<&'a Node>> {
        let mut levels = HashMap::new();
        Rtree::dfs(&self.root, &mut levels, 0);
        levels
    }

    fn dfs<'a>(r: &'a Node, levels: &mut HashMap<usize,Vec<&'a Node>>, h: usize) {
        levels.entry(h).or_insert(vec![]).push(r);
        match *r {
            Node::Leaf { tids: _ } => {},
            Node::Inner { mbrs: _, ref vs } => {
                for v in vs {
                    Rtree::dfs(v, levels, h + 1);
                }
            }
        }
    }
}

#[test]
fn mbr_test() {
    let table = Table {
        cols: vec![
            Column { xs: vec![1, 100, 20, 90, 1000] },
            Column { xs: vec![100, 20, 90, 1, 1000] },
        ],
    };

    let mbr = Rect::mbr_of(&[1, 3, 0], &table);
    let mbr_ex = Rect { 
        min: Point { xs: vec![1, 1] }, 
        max: Point { xs: vec![100, 100] },
        count: 3
    };
    assert_eq!(mbr, mbr_ex);

    let area_ex = ((100 - 1) as f32 * (100 - 1) as f32).ln();
    assert_eq!(mbr.area(), area_ex);

    let diag_ex = Point { xs: vec![100 - 1, 100 - 1] };
    assert_eq!(mbr.diag_vector(), diag_ex);

    let p_ex = Rect {
        min: Point { xs: vec![20, 90] },
        max: Point { xs: vec![20, 90] },
        count: 1
    };
    assert_eq!(Rect::of_p(&Point::of_tuple(2, &table)), p_ex);
}

#[test]
fn table_test() {
    {
        let table = Table { 
            cols: vec![
                Column { xs: vec![0, 100, 20, 90] },
                Column { xs: vec![0, 100, 20, 90] },
                Column { xs: vec![0, 100, 20, 90] },
            ],
        };
        assert_eq!(table.len(), 4);
        assert_eq!(table.width(), 3);
        assert!(!table.is_empty());
    }
    {
        let table = Table { cols: vec![] };
        assert_eq!(table.len(), 0);
        assert_eq!(table.width(), 0);
        assert!(table.is_empty());
    }
    {
        let table = Table { 
            cols: vec![
                Column { xs: vec![] },
            ],
        };
        assert_eq!(table.len(), 0);
        assert_eq!(table.width(), 1);
        assert!(table.is_empty());
    }
}

#[test]
fn column_test() {
    {
        let col = Column { xs: vec![0, 100, 20, 90] };
        assert!(!col.is_empty());
        assert_eq!(col.len(), col.xs.len());
        for (i, &x) in col.xs.iter().enumerate() {
            assert_eq!(col.val(i), x);
        }
    }
    {
        let col = Column { xs: vec![] };
        assert!(col.is_empty());
    }
}

#[test]
fn mbr_overlap_test() {
    for dim in 1..3 {
        let mut overlapped = [
            Rect { min: Point { xs: vec![10; dim] }, max: Point { xs: vec![20; dim] }, count: 1 },
            Rect { min: Point { xs: vec![15; dim] }, max: Point { xs: vec![25; dim] }, count: 2 },
            Rect { min: Point { xs: vec![10; dim] }, max: Point { xs: vec![10; dim] }, count: 3 },
        ];
        overlapped[1].min.xs[0] = 0; // [0, 15...15]
        overlapped[1].max.xs[0] = 15; // [15, 25...25]
        let mut nonoverlapped = [
            Rect { min: Point { xs: vec![10; dim] }, max: Point { xs: vec![20; dim] }, count: 1 },
            Rect { min: Point { xs: vec![0; dim]  }, max: Point { xs: vec![5; dim]  }, count: 2 },
        ];
        if dim > 1 {
            nonoverlapped[1].min.xs[0] = 20; // [20, 0..0]
            nonoverlapped[1].max.xs[0] = 30; // [30, 5...5]
        }
        assert!(overlapped[0].overlap(&overlapped[1]));
        assert!(overlapped[0].overlap(&overlapped[2]));
        assert!(!nonoverlapped[0].overlap(&nonoverlapped[1]));
    }
}

#[test]
fn phys_best_binary_split_test() {
    let table = Table { 
        cols: vec![
            Column { xs: vec![0, 100, 20, 90] },
            Column { xs: vec![0, 100, 20, 90] },
            Column { xs: vec![0, 100, 20, 90] },
        ],
    };
    let tids = (0..table.len()).collect::<Vec<TupleId>>();
    let orders = (0..table.width()).collect::<Vec<usize>>();
    let max_ps_per_subtree = 2;
    let (lpart,hpart) = Rtree::phys_best_binary_split(tids, &table, max_ps_per_subtree, &orders);
    assert_eq!(lpart, vec![0, 2]);
    assert_eq!(hpart, vec![3, 1]);
}

#[test]
fn phys_partition_test() {
    let num_ps = 7;
    let xs = shuffle(&(0..num_ps).map(|x| x * 10).collect::<Vec<Datum>>());
    let table = Table { 
        cols: vec![
            Column { xs: xs.clone() },
            Column { xs: xs.clone() },
            Column { xs: xs.clone() },
        ],
    };
    assert_eq!(table.len(), num_ps as usize);

    let max_ps_per_subtree = 2;
    let parts = Rtree::phys_partition(table.tids(), &table, max_ps_per_subtree, &table.orders());

    assert_eq!(parts.len(), (num_ps as f32 / max_ps_per_subtree as f32).ceil() as usize);
    assert_eq!(points_from(&parts[0], &table), vec![Point { xs: vec![0;  table.width()] }, Point { xs: vec![10; table.width()] }]); 
    assert_eq!(points_from(&parts[1], &table), vec![Point { xs: vec![20; table.width()] }, Point { xs: vec![30; table.width()] }]);
    assert_eq!(points_from(&parts[2], &table), vec![Point { xs: vec![40; table.width()] }, Point { xs: vec![50; table.width()] }]); 
    assert_eq!(points_from(&parts[3], &table), vec![Point { xs: vec![60; table.width()] }]);
}

fn shuffle(xs: &[Datum]) -> Vec<Datum> {
    let mut res = Vec::from(xs);
    {
        let _res = &mut res;
        let mut rng = rand::thread_rng();
        rng.shuffle(_res);
    }
    res
}

fn points_from(tids: &[TupleId], table: &Table) -> Vec<Point> {
    tids.iter().map(|&tid| Point::of_tuple(tid, table)).collect::<Vec<Point>>()
}

#[test]
fn simple_search_test() {
    let table = Table {
        cols: vec![
            Column { xs: (100..160).collect::<Vec<Datum>>() },
            Column { xs: (100..160).collect::<Vec<Datum>>() },
        ],
    };

    let root = Node::Inner {
        mbrs: vec![
            Rect { 
                min: Point { xs: vec![100, 100] },
                max: Point { xs: vec![130, 130] },
                count: 30,
            },
            Rect {
                min: Point { xs: vec![131, 131] },
                max: Point { xs: vec![160, 160] },
                count: 29,
            },
        ],
        vs: vec![
            Node::Leaf {
                tids: (1..31).collect::<Vec<TupleId>>(),
            },
            Node::Leaf {
                tids: (31..60).collect::<Vec<TupleId>>(),
            },
        ],
    };

    let tree = Rtree { root: root, table: table };

    assert_eq!((1..21).collect::<Vec<TupleId>>(), tree.box_search(&Rect { 
        min: Point { xs: vec![100, 100] },
        max: Point { xs: vec![120, 120] },
        count: 0,
    }));
}

#[test]
fn simple_rtree_test() {
    let xs = shuffle(&(100..160).collect::<Vec<Datum>>());
    let table = Table {
        cols: vec![
            Column { xs: xs.clone() },
            Column { xs: xs.clone() },
        ],
    };
    let phys_fanout =  3;
    let tree = Rtree::rtree_on_table(table.clone(), phys_fanout);
    let stats = tree.stats();
    assert_eq!(stats.get("table width"), Some(&table.width()));
    assert_eq!(stats.get("table length"), Some(&table.len()));
    assert_eq!(stats.get("tree height"), Some(&4));
    assert_eq!(stats.get("level 0"), Some(&1));
    assert_eq!(stats.get("level 1"), Some(&3));
    assert_eq!(stats.get("level 2"), Some(&7));
    assert_eq!(stats.get("level 3"), Some(&20));
    for tid in table.tids() {
        let expected = vec![tid];
        let actual = tree.box_search(&Rect::mbr_of(&expected, &table));
        assert_eq!(actual, expected);
    }
}

fn main() {
    let xs = shuffle(&(100..160).collect::<Vec<Datum>>());
    let table = Table {
        cols: vec![
            Column { xs: xs.clone() },
            Column { xs: xs.clone() },
        ],
    };
    let phys_fanout =  3;
    let tree = Rtree::rtree_on_table(table.clone(), phys_fanout);
    let stats = tree.stats();
    println!("tree stats: {:?}", stats);

    println!("tree: {:?}", tree.root);
}
