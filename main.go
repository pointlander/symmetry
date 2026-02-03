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

	"golang.org/x/crypto/sha3"
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

	LoadBooks()
}
