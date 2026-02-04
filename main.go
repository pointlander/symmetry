// Copyright 2026 The Symmetry Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"crypto/ed25519"
	"embed"
	"encoding/base32"
	"fmt"
	"hash/crc64"
	"io"
	"math"
	"math/rand"
	"strings"

	"golang.org/x/crypto/sha3"

	"github.com/pointlander/gradient/tf32"
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
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
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

// State is a state
type State struct {
	Shard
	Image []float32
}

// LearnEmbedding learns the embedding
func LearnEmbedding(rng *rand.Rand, buffer []State) {
	others := tf32.NewSet()
	others.Add("x", 256, len(buffer))
	x := others.ByName["x"]
	for i := range buffer {
		x.X = append(x.X, buffer[i].Image...)
	}

	set := tf32.NewSet()
	set.Add("i", 7, 128)
	set.Add("w0", 256, 256)
	set.Add("b0", 256)
	set.Add("w1", 2*256, 256)
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

// Embedding is a markov model
type Embedding struct {
	Model [Order]map[Markov][]float32
	Root  []float32
}

// NewEmbedding creates a new model
func NewEmbedding() (model Embedding) {
	for i := range Order {
		model.Model[i] = make(map[Markov][]float32)
	}
	model.Root = make([]float32, 256)
	return model
}

// Set sets an entry
func (m *Embedding) Set(markov Markov, entry, previous, next byte) {
	for i := range Order {
		bucket := m.Model[i][markov]
		if bucket == nil {
			bucket = make([]float32, 256)
		}
		bucket[previous]++
		bucket[next]++
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
	m.Root[entry]++
}

// Get gets an entry
func (m *Embedding) Get(markov Markov) []float32 {
	for i := range Order {
		bucket := m.Model[i][markov]
		if bucket != nil {
			return bucket
		}
		markov[i] = 0
	}
	return m.Root
}

func main() {
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
			sum := float32(0.0)
			for _, count := range value {
				sum += count
			}
			for j, count := range value {
				value[j] = count / sum
			}
		}
	}
	{
		sum := float32(0.0)
		for _, count := range embedding.Root {
			sum += count
		}
		for i, count := range embedding.Root {
			embedding.Root[i] = count / sum
		}
	}
	buffer := make([]State, 128)
	markov = Markov{}
	model := NewModel()
	for block := range 16 {
		fmt.Println("block", block)
		for i, symbol := range book.Text[128*block : 128*block+128] {
			markov.Iterate(symbol)
			embedding := embedding.Get(markov)
			buffer[i].Image = embedding
			buffer[i].Symbol = symbol
		}
		LearnEmbedding(rng, buffer)
		{
			markov := markov
			for i := range buffer {
				model.Set(markov, buffer[i])
				markov.Iterate(buffer[i].Symbol)
			}
		}
	}
}
