# TSP solver using neural nets

### Model

Number of input neurons = `2N`
Number of output neurons = `N`
where `N` is the number of locations.

**Mapping**

|Ordering|Encoding|Permutation|
| :------------: | :------------: | :------------: |
|1|0|2|
|2|0.5|3|
|3|1|1|

If the permutation is sorted, the encoding would be `[1, 0, 0.5]`. This means location 1 is at the end of permutation, given its encoding is higher. Location 2 is at the beginning as its encoding is 0. The higher the encoding, the further towards the end in the permutation. Each output neuron is related to its two input neurons.

E.g. If permutation is `[2, 3, 1]` where each number is the index of the location, the encoding outputs will be `[1, 0, 0.5]`.

### Data generation

`generate.py` will generate a data file consisting of randomly generated locations, normalized to be with in ranges between 0 and 1, and another file with the solved permutations. The output file does not encode the permutations. This has to be done in separately with `encode_data.py`

The locations data file is flattened to 1 dimension and looks like: `[x1, y1, x2, y2, x3, y3]`.

### Supervised learning

The model can be trained using the data generated from the generation step. The loss is easily calculated from the permutation encoding.

### Reinforcement learning

Coming soon. RL will enable the network to learn without a data generation step.

### Further work

Can ML be used to estimate the total length of the most efficient permutation? Also, need to improve the scalability - large problems sizes take a long time to train.
