// Copyright 2026 The Symmetry Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/ed25519"
	"encoding/base32"
	"fmt"
	"hash/crc64"
	"math"
	"math/rand"

	"golang.org/x/crypto/sha3"
)

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
}
