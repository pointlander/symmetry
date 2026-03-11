// Copyright 2026 The Symmetry Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"compress/bzip2"
	"crypto/ed25519"
	"embed"
	"encoding/base32"
	"encoding/csv"
	"flag"
	"fmt"
	"hash/crc64"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/crypto/sha3"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/symmetry/kmeans"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-2
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed books/*
var Books embed.FS

// Book is a book
type Book struct {
	Name string
	Text []byte
}

// LoadBooks loads books
func LoadBooks() []Book {
	books := []Book{
		{Name: "10.txt.utf-8.bz2"},
		{Name: "pg74.txt.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}
	load := func(book string) []byte {
		file, err := Books.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		return data
	}
	for i := range books {
		books[i].Text = load(fmt.Sprintf("books/%s", books[i].Name))
	}
	return books
}

// Reader is a random number generator
type Reader struct {
	Rng *rand.Rand
}

// Read generates some random numbers
func (r *Reader) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(r.Rng.Intn(256))
	}
	return len(p), nil
}

// Shard is a markov model entry
type Shard struct {
	Embedding []float32
	Symbol    byte
}

// Write writes a shard
func (s Shard) Write(output *os.File) {
	buffer := make([]byte, 4)
	for _, value := range s.Embedding {
		bits := math.Float32bits(value)
		for i := range buffer {
			buffer[i] = byte((bits >> (8 * i)) & 0xFF)
		}
		count, err := output.Write(buffer)
		if err != nil {
			panic(err)
		}
		if count != len(buffer) {
			panic("not all bytes written")
		}
	}
	buffer[0] = s.Symbol
	count, err := output.Write(buffer[:1])
	if err != nil {
		panic(err)
	}
	if count != 1 {
		panic("one byte was not written")
	}
}

// State is a state
type State struct {
	Shard
	Image []float32
}

const (
	// Width is the width of the model
	Width = 256
	// EmbeddingSize is the size of the embedding
	EmbeddingSize = 7
)

// LearnEmbedding learns the embedding
func LearnEmbedding(iterations int, rng *rand.Rand, buffer []State) float32 {
	others := tf32.NewSet()
	others.Add("x", Width, len(buffer))
	x := others.ByName["x"]
	for i := range buffer {
		x.X = append(x.X, buffer[i].Image...)
	}

	set := tf32.NewSet()
	set.Add("i", EmbeddingSize, len(buffer))

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float32, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}
	sa := tf32.T(tf32.Mul(tf32.Dropout(tf32.Square(set.Get("i")), dropout), tf32.T(others.Get("x"))))
	loss := tf32.Avg(tf32.Quadratic(others.Get("x"), sa))

	for iteration := range iterations {
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		set.Zero()
		others.Zero()
		l := tf32.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return l
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += float64(d * d)
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * float32(scaling)
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		//fmt.Println(iteration, l)
	}

	ii := set.ByName["i"]
	for i := range ii.S[1] {
		cp := make([]float32, ii.S[0])
		copy(cp, ii.X[i*ii.S[0]:(i+1)*ii.S[0]])
		buffer[i].Embedding = cp
	}

	{
		sa := tf32.T(tf32.Mul(tf32.Square(set.Get("i")), tf32.T(others.Get("x"))))
		loss := tf32.Avg(tf32.Quadratic(others.Get("x"), sa))
		set.Zero()
		others.Zero()
		return tf32.Gradient(loss).X[0]
	}
}

// Order is the order of the model
const Order = 4

// Markov is a markov state
type Markov [Order]byte

// Iterate iterates the markov state
func (m *Markov) Iterate(s byte) {
	for i := range m[:len(*m)-1] {
		m[i] = m[i+1]
	}
	m[len(*m)-1] = s
}

// Bucket is the entry in a markov model
type Bucket []Shard

// Model is a markov model
type Model struct {
	Model [Order]map[Markov]Bucket
	Root  Bucket
}

// NewModel creates a new model
func NewModel() (model Model) {
	for i := range model.Model {
		model.Model[i] = make(map[Markov]Bucket)
	}
	return model
}

// Set sets an entry
func (m *Model) Set(markov Markov, entry State) {
	for i := range Order {
		bucket := m.Model[i][markov]
		bucket = append(bucket, entry.Shard)
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
	m.Root = append(m.Root, entry.Shard)
}

// Get gets an entry
func (m *Model) Get(markov Markov) Bucket {
	for i := range Order {
		bucket := m.Model[i][markov]
		if bucket != nil {
			return bucket
		}
		markov[i] = 0
	}
	return m.Root
}

// Bucket is a markov bucket
type Entry struct {
	N  []float32
	N2 []float32
}

// Embedding is a markov model
type Embedding struct {
	Model [Order]map[Markov]Entry
	Root  []float32
}

// NewEmbedding creates a new model
func NewEmbedding() (model Embedding) {
	for i := range Order {
		model.Model[i] = make(map[Markov]Entry)
	}
	model.Root = make([]float32, 256)
	return model
}

// Set sets an entry
func (m *Embedding) Set(markov Markov, entry, previous, next byte) {
	for i := range Order {
		bucket := m.Model[i][markov]
		if bucket.N == nil {
			bucket.N = make([]float32, 256)
			bucket.N2 = make([]float32, 256)
		}
		bucket.N[previous]++
		bucket.N[next]++
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
	m.Root[entry]++
}

// SetNext sets an entry
func (m *Embedding) SetNext(markov Markov, entry, next, next2 byte) {
	for i := range Order {
		bucket := m.Model[i][markov]
		if bucket.N == nil {
			bucket.N = make([]float32, 256)
			bucket.N2 = make([]float32, 256)
		}
		bucket.N[next]++
		bucket.N2[next2]++
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
	m.Root[entry]++
}

// Get gets an entry
func (m *Embedding) Get(markov Markov) []float32 {
	for i := range Order {
		bucket := m.Model[i][markov]
		if bucket.N != nil {
			return bucket.N
		}
		markov[i] = 0
	}
	return m.Root
}

func cs(a, b []float32) float32 {
	ab := tf32.Dot(a, b)
	aa := tf32.Dot(a, a)
	bb := tf32.Dot(b, b)
	if aa <= 0 {
		return 0
	}
	if bb <= 0 {
		return 0
	}
	return ab / (float32(math.Sqrt(float64(aa))) * float32(math.Sqrt(float64(bb))))
}

// Walk is a walk on a model
type Walk struct {
	Symbols []Shard
	Cost    float32
}

func (model *Model) Walk(seed int64, markov Markov, current []float32, done chan Walk) {
	rng := rand.New(rand.NewSource(seed))
	symbols := make([]Shard, 0, 1024)
	cost := float32(0.0)
	context := make([]float32, len(current))
	copy(context, current)
	for range 1024 {
		bucket := model.Get(markov)
		sum := float32(0.0)
		d := make([]float32, len(bucket))
		for i, entry := range bucket {
			x := tf32.Dot(context, entry.Embedding)
			if x < 0 {
				x = -x
			}
			d[i] = x
			sum += x
		}
		total, selected, index := float32(0.0), rng.Float32(), 0
		for i := range bucket {
			total += d[i] / sum
			if selected < total {
				index = i
				break
			}
		}
		symbol := bucket[index]
		symbols = append(symbols, symbol)
		cost += d[index] / sum
		for i := range context {
			context[i] = (context[i] + bucket[index].Embedding[i])
		}
		factor := tf32.Dot(current, current)
		factor = float32(math.Sqrt(float64(factor)))
		for i, count := range current {
			current[i] = count / factor
		}
		markov.Iterate(symbol.Symbol)
	}
	done <- Walk{
		Symbols: symbols,
		Cost:    cost,
	}
}

// MarkovMode stores the vector symbol pairs in markov model
func MarkovMode() {
	data := []byte("andy")
	crc := crc64.Checksum(data, crc64.MakeTable(crc64.ISO))
	seed := int64(crc & math.MaxInt64)
	reader := Reader{
		Rng: rand.New(rand.NewSource(seed)),
	}
	publicKey, privateKey, err := ed25519.GenerateKey(&reader)
	if err != nil {
		panic(fmt.Errorf("Error generating keypair: %v", err))
	}

	fmt.Printf("Private Key (hex): %x\n", privateKey)
	fmt.Printf("Public Key (hex): %x\n", publicKey)

	version := byte(3)
	checksum := []byte(".onion checksum")
	checksum = append(checksum, publicKey...)
	checksum = append(checksum, version)
	check := sha3.Sum256(checksum)
	onion_address := []byte{}
	onion_address = append(onion_address, publicKey...)
	onion_address = append(onion_address, check[:2]...)
	onion_address = append(onion_address, version)
	address := base32.StdEncoding.EncodeToString(onion_address) + ".onion"
	fmt.Println(address)

	books := LoadBooks()
	book := books[1]
	fmt.Println("length", len(book.Text))
	embedding := NewEmbedding()
	markov := Markov{}
	var previous byte
	for i, value := range book.Text[:len(book.Text)-1] {
		markov.Iterate(value)
		next := book.Text[i+1]
		embedding.Set(markov, value, previous, next)
		previous = value
	}

	rng := rand.New(rand.NewSource(1))
	for i := range embedding.Model {
		for _, value := range embedding.Model[i] {
			factor := tf32.Dot(value.N, value.N)
			if factor <= 0 {
				continue
			}
			factor = float32(math.Sqrt(float64(factor)))
			for j, count := range value.N {
				value.N[j] = count / factor
			}
		}
	}
	{
		factor := tf32.Dot(embedding.Root, embedding.Root)
		if factor > 0 {
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range embedding.Root {
				embedding.Root[i] = count / factor
			}
		}
	}
	buffer := make([]State, 128)
	markov = Markov{}
	model := NewModel()
	for block := range 1024 {
		fmt.Println("block", block)
		save := markov
		for i, symbol := range book.Text[128*block : 128*block+128] {
			markov.Iterate(symbol)
			embedding := embedding.Get(markov)
			buffer[i].Image = embedding
			buffer[i].Symbol = symbol
		}
		LearnEmbedding(256, rng, buffer)
		for i := range buffer {
			model.Set(save, buffer[i])
			save.Iterate(buffer[i].Symbol)
		}
	}

	{
		prompt := []byte("What is the meaning of life?")
		markov := Markov{}
		buffer := make([]State, len(prompt))
		for i, symbol := range prompt {
			markov.Iterate(symbol)
			embedding := embedding.Get(markov)
			buffer[i].Image = embedding
			buffer[i].Symbol = symbol
		}
		var current []float32
		for {
			//LearnEmbedding(256, rng, buffer)
			if current == nil {
				LearnEmbedding(256, rng, buffer)
				current = make([]float32, EmbeddingSize)
				copy(current, buffer[0].Embedding)
				for _, entry := range buffer {
					for i := range current {
						current[i] = (current[i] + entry.Embedding[i])
					}
					factor := tf32.Dot(current, current)
					factor = float32(math.Sqrt(float64(factor)))
					for i, count := range current {
						current[i] = count / factor
					}
				}
			} /*else {
				for i := range current {
					current[i] = (current[i] + buffer[len(buffer)-1].Embedding[i])
				}
				factor := tf32.Dot(current, current)
				factor = float32(math.Sqrt(float64(factor)))
				for i, count := range current {
					current[i] = count / factor
				}
			}*/
			results := make([]Walk, 0, 8*1024)
			i, flight, done, cpus := 0, 0, make(chan Walk, 8), runtime.NumCPU()
			for i < 8*1024 && flight < cpus {
				go model.Walk(rng.Int63(), markov, current, done)
				flight++
				i++
			}
			for i < 8*1024 {
				results = append(results, <-done)
				flight--

				go model.Walk(rng.Int63(), markov, current, done)
				flight++
				i++
			}
			for range flight {
				results = append(results, <-done)
			}
			sort.Slice(results, func(i, j int) bool {
				return results[i].Cost > results[j].Cost
			})
			index := 0
			sum := float32(0.0)
			for i := range results[:33] {
				sum += results[i].Cost
			}
			total, selected, index := float32(0.0), rng.Float32(), 0
			for i := range results[:33] {
				total += results[i].Cost / sum
				if selected < total {
					index = i
					break
				}
			}
			symbol := results[index].Symbols[0].Symbol
			fmt.Printf("%c", symbol)
			for i := range current {
				current[i] = (current[i] + results[index].Symbols[0].Embedding[i])
			}
			factor := tf32.Dot(current, current)
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range current {
				current[i] = count / factor
			}
			markov.Iterate(symbol)
			/*embedding := embedding.Get(markov)
			if len(buffer) < 33 {
				buffer = append(buffer, State{
					Image: embedding,
					Shard: Shard{
						Symbol: symbol,
					},
				})
			} else {
				for i := range buffer[:len(buffer)-1] {
					buffer[i] = buffer[i+1]
				}
				buffer[len(buffer)-1] = State{
					Image: embedding,
					Shard: Shard{
						Symbol: symbol,
					},
				}
			}*/
		}
	}
}

// Level is a tree level
type Level struct {
	Output *os.File
	Shards []Shard
}

func (l Level) Read(index int) Shard {
	_, err := l.Output.Seek(int64(index*(1+4*EmbeddingSize)), 0)
	if err != nil {
		panic(err)
	}
	shard := Shard{
		Embedding: make([]float32, EmbeddingSize),
	}
	buffer := make([]byte, 4)
	for i := range shard.Embedding {
		count, err := l.Output.Read(buffer)
		if err != nil {
			panic(err)
		}
		if count != len(buffer) {
			panic("not all bytes written")
		}
		bits := uint32(0)
		for j := range buffer {
			bits |= uint32(buffer[j]) << (8 * j)
		}
		shard.Embedding[i] = math.Float32frombits(bits)
	}
	count, err := l.Output.Read(buffer[:1])
	if err != nil {
		panic(err)
	}
	if count != 1 {
		panic("one byte was not written")
	}
	shard.Symbol = buffer[0]
	return shard
}

// Tree is a tree
type Tree struct {
	Levels []Level
}

// NewTree loads a tree
func NewTree() Tree {
	entries, err := os.ReadDir("./tree")
	if err != nil {
		panic(err)
	}
	max := 0
	for _, entry := range entries {
		parts := strings.Split(entry.Name(), ".")
		if len(parts) != 2 {
			panic("there should be two parts")
		}
		level, err := strconv.Atoi(parts[0])
		if err != nil {
			panic("part should be int")
		}
		if level > max {
			max = level
		}
	}
	levels := make([]Level, max+1)
	for _, entry := range entries {
		parts := strings.Split(entry.Name(), ".")
		if len(parts) != 2 {
			panic("there should be two parts")
		}
		level, err := strconv.Atoi(parts[0])
		if err != nil {
			panic("part should be int")
		}
		input, err := os.Open("./tree/" + entry.Name())
		if err != nil {
			panic(err)
		}
		levels[level].Output = input
	}
	return Tree{
		Levels: levels,
	}
}

// Lookup looks a shard up
func (t Tree) Lookup(rng *rand.Rand, vector []float32) (Shard, float32) {
	left, right := 0, 1
	shard := Shard{}
	cost := float32(0)
	for l := len(t.Levels) - 2; l >= 0; l-- {
		ll := t.Levels[l].Read(left)
		rr := t.Levels[l].Read(right)
		a := cs(ll.Embedding, vector)
		if a < 0 {
			a = -a
		}
		b := cs(rr.Embedding, vector)
		if b < 0 {
			b = -b
		}
		sum := a + b
		if rng.Float32() < a/sum {
			left, right = left*2, left*2+1
			shard = ll
			cost += a / sum
		} else {
			left, right = right*2, right*2+1
			shard = rr
			cost += b / sum
		}
	}
	return shard, cost
}

// Job is a walk job
type Job struct {
	Walk
	T *Tree
}

// Walk performs mcts
func (t *Tree) Walk(seed int64, current []float32, done chan Job) {
	rng := rand.New(rand.NewSource(seed))
	symbols := make([]Shard, 0, 1024)
	cost := float32(0.0)
	context := make([]float32, len(current))
	copy(context, current)
	for range 1024 {
		symbol, d := t.Lookup(rng, current)
		symbols = append(symbols, symbol)
		cost += d
		for i := range context {
			context[i] = (context[i] + symbol.Embedding[i])
		}
		factor := tf32.Dot(current, current)
		factor = float32(math.Sqrt(float64(factor)))
		for i, count := range current {
			current[i] = count / factor
		}
	}
	done <- Job{
		Walk: Walk{
			Symbols: symbols,
			Cost:    cost,
		},
		T: t,
	}
}

// Close closes all of the tree files
func (t Tree) Close() {
	for i := range t.Levels {
		t.Levels[i].Output.Close()
	}
}

// TreeMode stores the vector symbol pairs in aa tree
func TreeMode() {
	books := LoadBooks()
	book := books[1]
	fmt.Println("length", len(book.Text))
	embedding := NewEmbedding()
	markov := Markov{}
	var previous byte
	for i, value := range book.Text[:len(book.Text)-1] {
		markov.Iterate(value)
		next := book.Text[i+1]
		embedding.Set(markov, value, previous, next)
		previous = value
	}

	rng := rand.New(rand.NewSource(1))
	for i := range embedding.Model {
		for _, value := range embedding.Model[i] {
			factor := tf32.Dot(value.N, value.N)
			if factor <= 0 {
				continue
			}
			factor = float32(math.Sqrt(float64(factor)))
			for j, count := range value.N {
				value.N[j] = count / factor
			}
		}
	}
	{
		factor := tf32.Dot(embedding.Root, embedding.Root)
		if factor > 0 {
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range embedding.Root {
				embedding.Root[i] = count / factor
			}
		}
	}

	if *FlagPrompt != "" {
		markov := Markov{}
		prompt := []byte(*FlagPrompt)
		buffer := make([]State, len(prompt))
		for i, symbol := range prompt {
			markov.Iterate(symbol)
			embedding := embedding.Get(markov)
			buffer[i].Image = embedding
			buffer[i].Symbol = symbol
		}
		var current []float32
		LearnEmbedding(256, rng, buffer)
		current = make([]float32, EmbeddingSize)
		copy(current, buffer[0].Embedding)
		for _, entry := range buffer {
			for i := range current {
				current[i] = (current[i] + entry.Embedding[i])
			}
			factor := tf32.Dot(current, current)
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range current {
				current[i] = count / factor
			}
		}
		trees := make([]Tree, runtime.NumCPU())
		for i := range trees {
			trees[i] = NewTree()
		}
		defer func() {
			for i := range trees {
				trees[i].Close()
			}
		}()
		fmt.Println(len(trees[0].Levels))
		fmt.Println(trees[0].Lookup(rng, current))
		for {
			results := make([]Walk, 0, 8*1024)
			i, flight, done, cpus := 0, 0, make(chan Job, 8), runtime.NumCPU()
			for i < 256 && flight < cpus {
				go trees[flight].Walk(rng.Int63(), current, done)
				flight++
				i++
			}
			for i < 256 {
				result := <-done
				results = append(results, result.Walk)
				flight--

				go result.T.Walk(rng.Int63(), current, done)
				flight++
				i++
			}
			for range flight {
				result := <-done
				results = append(results, result.Walk)
			}
			sort.Slice(results, func(i, j int) bool {
				return results[i].Cost > results[j].Cost
			})
			index := 0
			sum := float32(0.0)
			for i := range results[:33] {
				sum += results[i].Cost
			}
			total, selected, index := float32(0.0), rng.Float32(), 0
			for i := range results[:33] {
				total += results[i].Cost / sum
				if selected < total {
					index = i
					break
				}
			}
			symbol := results[index].Symbols[0].Symbol
			fmt.Printf("%c", symbol)
			for i := range current {
				current[i] = (current[i] + results[index].Symbols[0].Embedding[i])
			}
			factor := tf32.Dot(current, current)
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range current {
				current[i] = count / factor
			}
		}
	}

	buffer := make([]State, 128)
	markov = Markov{}
	tree := make([]Level, 1, 8)
	var err error
	tree[0].Output, err = os.Create("tree/0.bin")
	if err != nil {
		panic(err)
	}
	defer func() {
		for i := range tree {
			tree[i].Output.Close()
		}
	}()
	for block := range 1024 {
		fmt.Println("block", block)
		for i, symbol := range book.Text[128*block : 128*block+128] {
			markov.Iterate(symbol)
			embedding := embedding.Get(markov)
			buffer[i].Image = embedding
			buffer[i].Symbol = symbol
		}
		LearnEmbedding(256, rng, buffer)
		for i := range buffer {
			tree[0].Shards = append(tree[0].Shards, buffer[i].Shard)
			for j := range tree {
				if len(tree[j].Shards)%2 == 0 && len(tree[j].Shards) > 1 {
					vector := make([]float32, EmbeddingSize)
					offset := len(tree[j].Shards) - 2
					for k := range tree[j].Shards[offset].Embedding {
						vector[k] = tree[j].Shards[offset].Embedding[k] +
							tree[j].Shards[offset+1].Embedding[k]
					}
					done := false
					if j+1 >= len(tree) {
						done = true
						output, err := os.Create(fmt.Sprintf("tree/%d.bin", j+1))
						if err != nil {
							panic(err)
						}
						tree = append(tree, Level{
							Output: output,
						})
					}
					tree[j+1].Shards = append(tree[j+1].Shards, Shard{
						Embedding: vector,
					})
					if done {
						break
					}
				} else {
					break
				}
			}
		}
	}
	for i := range tree {
		for j := range tree[i].Shards {
			tree[i].Shards[j].Write(tree[i].Output)
		}
	}
}

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures  []float64
	Embedding []float64
	Label     string
	L         byte
	Cluster   int
	Index     int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
	"gen":             3,
}

// Inverse is the labels inverse map
var Inverse = [4]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
	"gen",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// LearnEmbeddingIris learns the iris embedding
func LearnEmbeddingIris(iris []Fisher, size, width, iterations int) []Fisher {
	const Eta = 1e-3
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	length := len(iris)
	cp := make([]Fisher, length)
	copy(cp, iris)
	others.Add("x", size, len(cp))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}

	set := tf64.NewSet()
	set.Add("i", width, len(cp))

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor*.01)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(others.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(others.Get("x"), sa))

	for iteration := range iterations {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				_ = mhat
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				/*if rng.Float64() > .01 {
					w.X[ii] -= .05 * g
				} else {
					w.X[ii] += .05 * g
				}*/
			}
		}
		fmt.Println(l)
	}

	meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, width)
			for ii := range row {
				row[ii] = y.X[i*width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	for _, value := range x.X[len(iris)*size:] {
		cp[len(iris)].Measures = append(cp[len(iris)].Measures, value)
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
	}
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})
	return cp
}

// ClusterMode is the iris clustering mode
func ClusterMode() {
	rng := rand.New(rand.NewSource(1))
	iris := Load()
	rng.Shuffle(len(iris), func(i, j int) {
		iris[i], iris[j] = iris[j], iris[i]
	})
	cp5 := LearnEmbeddingIris(iris, 4, 5, 2*1024)
	iris2 := make([]Fisher, len(cp5))
	copy(iris2, cp5)
	/*dot := func(a, b []float64) float64 {
		x := 0.0
		for i, value := range a {
			x += value * b[i]
		}
		return x
	}*/
	for i := range iris2 {
		iris2[i].Measures = append(iris2[i].Measures, iris2[i].Embedding...)
		/*factor := dot(iris2[i].Measures, iris2[i].Measures)
		factor = math.Sqrt(factor)
		for j := range iris2[i].Measures {
			iris2[i].Measures[j] /= factor
		}*/
	}
	cp52 := LearnEmbeddingIris(iris2, 9, 5, 2*1024)
	acc5 := make(map[string][4]int)
	for i := range cp5 {
		fmt.Println(cp5[i].Cluster, cp5[i].Label)
		counts := acc5[cp5[i].Label]
		counts[cp5[i].Cluster]++
		acc5[cp5[i].Label] = counts
	}
	acc52 := make(map[string][4]int)
	for i := range cp52 {
		fmt.Println(cp52[i].Cluster, cp52[i].Label)
		counts := acc52[cp52[i].Label]
		counts[cp52[i].Cluster]++
		acc52[cp52[i].Label] = counts
	}

	for i, v := range acc5 {
		fmt.Println(i, v)
	}
	for i, v := range acc52 {
		fmt.Println(i, v)
	}
}

const (
	// EmbeddingWidth embedding width
	EmbeddingWidth = 5
)

// LearnEmbeddingBlock learns the embedding
func LearnEmbeddingBlock(rng *rand.Rand, buffer []State) {
	others := tf32.NewSet()
	others.Add("x", Width+EmbeddingWidth, len(buffer))
	x := others.ByName["x"]
	for i := range buffer {
		x.X = append(x.X, buffer[i].Image...)
	}

	set := tf32.NewSet()
	set.Add("i", EmbeddingWidth, len(buffer))

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float32, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}
	sa := tf32.T(tf32.Mul(tf32.Dropout(tf32.Square(set.Get("i")), dropout), tf32.T(others.Get("x"))))
	loss := tf32.Avg(tf32.Quadratic(others.Get("x"), sa))

	for iteration := range 256 {
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		set.Zero()
		others.Zero()
		l := tf32.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += float64(d * d)
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * float32(scaling)
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		//fmt.Println(iteration, l)
	}

	ii := set.ByName["i"]
	for i := range ii.S[1] {
		cp := make([]float32, ii.S[0])
		copy(cp, ii.X[i*ii.S[0]:(i+1)*ii.S[0]])
		buffer[i].Embedding = cp
	}
}

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e38*math.SmallestNonzeroFloat32
)

func softmax(values []float32, T float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			v /= T
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = exp(value/T - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// BlockMode block mode
func BlockMode() {
	books := LoadBooks()
	book := books[1]
	fmt.Println("length", len(book.Text))
	embedding := NewEmbedding()
	markov := Markov{}
	var previous byte
	for i, value := range book.Text[:len(book.Text)-1] {
		markov.Iterate(value)
		next := book.Text[i+1]
		embedding.Set(markov, value, previous, next)
		previous = value
	}
	for i := range embedding.Model {
		for _, value := range embedding.Model[i] {
			factor := tf32.Dot(value.N, value.N)
			if factor <= 0 {
				continue
			}
			factor = float32(math.Sqrt(float64(factor)))
			for j, count := range value.N {
				value.N[j] = count / factor
			}
		}
	}
	{
		factor := tf32.Dot(embedding.Root, embedding.Root)
		if factor > 0 {
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range embedding.Root {
				embedding.Root[i] = count / factor
			}
		}
	}

	rng := rand.New(rand.NewSource(1))
	set := tf32.NewSet()
	set.Add("w0", EmbeddingWidth, 32)
	set.Add("b0", 32)
	set.Add("w1", 64, 256)
	set.Add("b1", 256)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float32, len(w.X))
		}
	}

	others := tf32.NewSet()
	others.Add("input", EmbeddingWidth)
	others.Add("output", 256)
	input := others.ByName["input"]
	input.X = input.X[:cap(input.X)]
	output := others.ByName["output"]
	output.X = output.X[:cap(output.X)]

	l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
	loss := tf32.Quadratic(l1, others.Get("output"))

	buffer := make([]State, 16)
	for i := range buffer {
		buffer[i].Image = make([]float32, Width+EmbeddingWidth)
	}
	markov = Markov{}
	for iteration, value := range book.Text[:4*1024] {
		markov.Iterate(value)
		image := embedding.Get(markov)
		copy(buffer[0].Image, image)
		LearnEmbeddingBlock(rng, buffer)
		sum := make([]float32, EmbeddingWidth)
		for i := range buffer {
			for j := range sum {
				sum[j] += buffer[i].Embedding[j]
			}
			factor := tf32.Dot(buffer[i].Embedding, buffer[i].Embedding)
			factor = float32(math.Sqrt(float64(factor)))
			for j := range buffer[i].Embedding {
				buffer[i].Embedding[j] /= factor
			}
			copy(buffer[i].Image[Width:], buffer[i].Embedding)
		}
		factor := tf32.Dot(sum, sum)
		factor = float32(math.Sqrt(float64(factor)))
		for i := range sum {
			sum[i] /= factor
		}

		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		set.Zero()
		others.Zero()
		copy(input.X, sum)
		for j := range output.X {
			output.X[j] = 0
		}
		output.X[book.Text[iteration+1]] = 1
		l := tf32.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += float64(d * d)
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * float32(scaling)
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		fmt.Println(iteration, l)
	}

	markov = Markov{}
	prompt := []byte("What is the meaning of life?")
	out := []byte{}
	symbol := byte(0)
	for _, value := range prompt {
		markov.Iterate(value)
		image := embedding.Get(markov)
		copy(buffer[0].Image, image)
		LearnEmbeddingBlock(rng, buffer)
		sum := make([]float32, EmbeddingWidth)
		for i := range buffer {
			for j := range sum {
				sum[j] += buffer[i].Embedding[j]
			}
			factor := tf32.Dot(buffer[i].Embedding, buffer[i].Embedding)
			factor = float32(math.Sqrt(float64(factor)))
			for j := range buffer[i].Embedding {
				buffer[i].Embedding[j] /= factor
			}
			copy(buffer[i].Image[Width:], buffer[i].Embedding)
		}
		factor := tf32.Dot(sum, sum)
		factor = float32(math.Sqrt(float64(factor)))
		for i := range sum {
			sum[i] /= factor
		}

		set.Zero()
		others.Zero()
		copy(input.X, sum)
		l1(func(a *tf32.V) bool {
			softmax(a.X, .5)
			total, selected := float32(0.0), rng.Float32()
			for i, value := range a.X {
				total += value
				if selected < total {
					symbol = byte(i)
					break
				}
			}
			return true
		})
		out = append(out, symbol)
	}
	fmt.Println(string(out))
	for {
		markov.Iterate(symbol)
		image := embedding.Get(markov)
		copy(buffer[0].Image, image)
		LearnEmbeddingBlock(rng, buffer)
		sum := make([]float32, EmbeddingWidth)
		for i := range buffer {
			for j := range sum {
				sum[j] += buffer[i].Embedding[j]
			}
			factor := tf32.Dot(buffer[i].Embedding, buffer[i].Embedding)
			factor = float32(math.Sqrt(float64(factor)))
			for j := range buffer[i].Embedding {
				buffer[i].Embedding[j] /= factor
			}
			copy(buffer[i].Image[Width:], buffer[i].Embedding)
		}
		factor := tf32.Dot(sum, sum)
		factor = float32(math.Sqrt(float64(factor)))
		for i := range sum {
			sum[i] /= factor
		}

		set.Zero()
		others.Zero()
		copy(input.X, sum)
		l1(func(a *tf32.V) bool {
			softmax(a.X, .5)
			total, selected := float32(0.0), rng.Float32()
			for i, value := range a.X {
				total += value
				if selected < total {
					symbol = byte(i)
					break
				}
			}
			return true
		})
		fmt.Printf("%c", symbol)
	}
}

var (
	// FlagMarkov is the markov mode
	FlagMarkov = flag.Bool("markov", false, "markov mode")
	// FlagTree is the tree mode
	FlagTree = flag.Bool("tree", false, "tree mode")
	// FlagCluster cluster mode
	FlagCluster = flag.Bool("cluster", false, "cluster mode")
	// FlagPrompt inference mode
	FlagPrompt = flag.String("prompt", "", "inference mode")
	// FlagBlock
	FlagBlock = flag.Bool("block", false, "block mode")
)

func main() {
	flag.Parse()

	if *FlagMarkov {
		MarkovMode()
		return
	}

	if *FlagTree || *FlagPrompt != "" {
		TreeMode()
		return
	}

	if *FlagCluster {
		ClusterMode()
		return
	}

	if *FlagBlock {
		BlockMode()
		return
	}

	rng := rand.New(rand.NewSource(1))

	books := LoadBooks()
	embedding := NewEmbedding()
	for _, book := range books {
		fmt.Println("length", len(book.Text))
		markov := Markov{}
		for i, value := range book.Text[:len(book.Text)-2] {
			markov.Iterate(value)
			embedding.SetNext(markov, value, book.Text[i+1], book.Text[i+2])
		}
	}
	for i := range embedding.Model {
		for _, value := range embedding.Model[i] {
			factor := tf32.Dot(value.N, value.N)
			if factor <= 0 {
				continue
			}
			factor = float32(math.Sqrt(float64(factor)))
			for j, count := range value.N {
				value.N[j] = count / factor
			}
		}
	}
	{
		factor := tf32.Dot(embedding.Root, embedding.Root)
		if factor > 0 {
			factor = float32(math.Sqrt(float64(factor)))
			for i, count := range embedding.Root {
				embedding.Root[i] = count / factor
			}
		}
	}

	type Result struct {
		Value []byte
		S     float32
	}
	results := make([]Result, 0, 127)
	for range 256 {
		markov := Markov{}
		buffer := make([]State, 0, 8)
		prompt := []byte("What is the meaning of life?")
		for _, value := range prompt {
			markov.Iterate(value)
			buffer = append(buffer, State{
				Image: embedding.Get(markov),
			})
		}
		for range 33 {
			distribution := embedding.Get(markov)
			sum := float32(0.0)
			for _, value := range distribution {
				sum += value
			}
			total, selected, index := float32(0.0), rng.Float32(), 0
			for i, value := range distribution {
				total += value / sum
				if selected < total {
					index = i
					break
				}
			}
			prompt = append(prompt, byte(index))
			markov.Iterate(byte(index))
			buffer = append(buffer, State{
				Image: embedding.Get(markov),
			})
		}
		LearnEmbedding(512, rng, buffer)
		s := float32(0.0)
		for i := range EmbeddingSize {
			sum := float32(0.0)
			for j := range buffer {
				sum += buffer[j].Embedding[i]
			}
			avg := sum / float32(len(buffer))
			variance := float32(0.0)
			for j := range buffer {
				diff := buffer[j].Embedding[i] - avg
				variance += diff * diff
			}
			variance /= float32(len(buffer))
			s += float32(math.Sqrt(float64(variance)))
		}
		results = append(results, Result{
			Value: prompt,
			S:     s,
		})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].S > results[j].S
	})
	for _, value := range results {
		fmt.Println(value.S, string(value.Value))
	}
}
