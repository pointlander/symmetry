/*
Package pagerank implements the *weighted* PageRank algorithm.
*/
package pagerank

import (
	"math"
	"math/rand"
)

type node struct {
	weight   float64
	outbound float64
}

// Graph holds node and edge data.
type Graph struct {
	rng   *rand.Rand
	edges [][]float64
	nodes []*node
}

// NewGraph initializes and returns a new graph.
func NewGraph(count int, rng *rand.Rand) *Graph {
	return &Graph{
		rng:   rng,
		edges: make([][]float64, count),
		nodes: make([]*node, count),
	}
}

// Link creates a weighted edge between a source-target node pair.
// If the edge already exists, the weight is incremented.
func (self *Graph) Link(source, target uint32, weight float64) {
	if n := self.nodes[source]; n == nil {
		self.nodes[source] = &node{
			weight:   0,
			outbound: 0,
		}
	}

	self.nodes[source].outbound += weight

	if n := self.nodes[target]; n == nil {
		self.nodes[target] = &node{
			weight:   0,
			outbound: 0,
		}
	}

	if e := self.edges[source]; e == nil {
		self.edges[source] = make([]float64, len(self.edges))
	}

	self.edges[source][target] += weight
}

// Rank computes the PageRank of every node in the directed graph.
// α (alpha) is the damping factor, usually set to 0.85.
// ε (epsilon) is the convergence criteria, usually set to a tiny value.
//
// This method will run as many iterations as needed, until the graph converges.
func (self *Graph) Rank(α, ε float64, callback func(id int, rank float64)) {
	Δ := float64(1.0)
	inverse := 1 / float64(len(self.nodes))

	// Normalize all the edge weights so that their sum amounts to 1.
	for source := range self.edges {
		if self.nodes[source] != nil && self.nodes[source].outbound > 0 {
			for target := range self.edges[source] {
				self.edges[source][target] /= self.nodes[source].outbound
			}
		}
	}

	for key := range self.nodes {
		if self.nodes[key] == nil {
			continue
		}
		self.nodes[key].weight = inverse
	}

	for Δ > ε {
		leak := float64(0)
		nodes := make([]float64, len(self.nodes))
		perm := self.rng.Perm(len(self.nodes))

		for key := range perm {
			value := self.nodes[key]
			if value == nil {
				continue
			}
			nodes[key] = value.weight

			if value.outbound == 0 {
				leak += value.weight
			}

			self.nodes[key].weight = 0
		}

		leak *= α

		for source := range perm {
			for target, weight := range self.edges[source] {
				if self.nodes[target] == nil {
					continue
				}
				self.nodes[target].weight += α * nodes[source] * weight
			}
			if self.nodes[source] == nil {
				continue
			}
			self.nodes[source].weight += (1-α)*inverse + leak*inverse
		}

		Δ = 0

		for key, value := range self.nodes {
			if value == nil {
				continue
			}
			Δ += math.Abs(value.weight - nodes[key])
		}
	}

	for key, value := range self.nodes {
		if value == nil {
			continue
		}
		callback(key, value.weight)
	}
}

// Reset clears all the current graph data.
func (self *Graph) Reset(count int) {
	self.edges = make([][]float64, count)
	self.nodes = make([]*node, count)
}
