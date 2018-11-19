Distributed map reduce with GPU acceleration
======================
**Authors:** Edward Atter, Siyu Zheng, Yan Wu

## Overview
Map reduce is an important algorithm in the age of big data. It serves as a template, allowing other programmers to efficiently build a program capable of processing enourmous amounts of data using the power of distriuted computing. The basic example is word count, but it's applications expand much further to other well known algorithms such as Google's PageRank and friend recommendation systems. 

Traditionally, this is accomplished on a CPU using software such as Hadoop. To achieve true paralellism, thousands of computers are often used together in a cluster. By utilizing the native paralellism present in GPU architecture, we theorize many less computers can be clustered together to yield the same performance; resulting in a much better performance per dollar. 

A quote from NetworkWorld about distributed MapReduce on a GPU explains, "most of [the frameworks] are no longer supported and were built for particular scientific projects." This is the need we hope to fulfill.

## Milestones
#### 11/19

- Basic map reduce on single GPU
- Word count example

#### 11/26

- Map reduce working distributed across 2 different computers
- PageRank or similar algorithm that has more work per thread than word count

#### 12/03

- Scales to 3+ computers
- Measure network traffic / estimate network bottleneck
- Shared memory

#### 12/10

- Deep dive into efficiency
- Pegged memory
- Other advanced CUDA topics
- Compare with traditional MapReduce programs eg. Hadoop
