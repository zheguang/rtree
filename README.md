# rtree
A Rust implementation of [R-Tree](https://en.wikipedia.org/wiki/R-tree) on column-oriented format.

## Features
1. Both the R-Tree and the table reside in main memory.  Future works include extension to secondary storage.
2. Search tuples by [hypercube](https://en.wikipedia.org/wiki/Hypercube).  
3. Bulkload from a table.  Future work include per-tuple update/delete/insert.

## Build
```bash
cargo build
cargo test
```

## Run
```bash
cargo run
```
