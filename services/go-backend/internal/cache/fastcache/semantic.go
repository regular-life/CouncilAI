package fastcache

// #cgo CXXFLAGS: -std=c++17 -O3 -mavx2
// #include "semantic_cache.h"
// #include <stdlib.h>
import "C"

import (
	"encoding/json"
	"fmt"
	"log"
	"unsafe"
)

type SemanticCache struct {
	handle C.SemanticCacheHandle
}

func NewSemanticCache(capacity int) *SemanticCache {
	return &SemanticCache{
		handle: C.SemanticCache_Create(C.int(capacity)),
	}
}

func (s *SemanticCache) Destroy() {
	if s.handle != nil {
		C.SemanticCache_Destroy(s.handle)
		s.handle = nil
	}
}

func (s *SemanticCache) Put(docID string, vector []float32, response interface{}) error {
	if len(vector) != 384 {
		return fmt.Errorf("vector dimension mismatch: expected 384, got %d", len(vector))
	}

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to marshal cache response: %w", err)
	}

	cDocID := C.CString(docID)
	cAnswer := C.CString(string(data))
	defer C.free(unsafe.Pointer(cDocID))
	defer C.free(unsafe.Pointer(cAnswer))

	cVector := (*C.float)(&vector[0])

	C.SemanticCache_Put(s.handle, cDocID, cVector, cAnswer)
	log.Printf("[FastCache] Put doc_id=%s, vector_dim=384", docID)
	return nil
}

func (s *SemanticCache) Get(docID string, vector []float32, threshold float32, dest interface{}) bool {
	if len(vector) != 384 {
		return false
	}

	cDocID := C.CString(docID)
	defer C.free(unsafe.Pointer(cDocID))

	cVector := (*C.float)(&vector[0])
	cThreshold := C.float(threshold)

	cResult := C.SemanticCache_Get(s.handle, cDocID, cVector, cThreshold)
	if cResult == nil {
		return false
	}
	defer C.free(unsafe.Pointer(cResult))

	resultStr := C.GoString(cResult)
	if err := json.Unmarshal([]byte(resultStr), dest); err != nil {
		log.Printf("[FastCache] json unmarshal failed: %v", err)
		return false
	}
	return true
}
