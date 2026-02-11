// Copyright 2025 The Snow Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
)

func sqrt(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}

func exp(a float32) float32 {
	return float32(math.Exp(float64(a)))
}

func log(a float32) float32 {
	return float32(math.Log(float64(a)))
}
