"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

## Testing addition
x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))


## Testing Multiplication
g = Mul(x, y, z)
feed_dict = {x: 4, y: 5, z: 10}
graph = topological_sort(feed_dict)
output = forward_pass(g, graph)
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))


## Testing addition with weights (Linear)
inputs, weights, bias = Input(), Input(), Input()

h = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(h, graph)

print(output) # should be 12.7 with this example





