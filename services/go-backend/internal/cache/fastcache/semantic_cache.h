#ifndef SEMANTIC_CACHE_H
#define SEMANTIC_CACHE_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to hide C++ class implementation from Go
typedef void* SemanticCacheHandle;

// Create a new semantic cache instance with the given LRU capacity
SemanticCacheHandle SemanticCache_Create(int capacity);

// Destroy the cache instance to free memory
void SemanticCache_Destroy(SemanticCacheHandle cache);

// Insert an entry into the cache
// doc_id: Document ID namespace
// vector_data: Pointer to array of 384 floats
// answer: The JSON response string
void SemanticCache_Put(SemanticCacheHandle cache, const char* doc_id, const float* vector_data, const char* answer);

// Fetch an entry from the cache
// If an entry is found with a cosine similarity >= threshold, returns a malloc'd string containing the answer.
// The caller is responsible for freeing the exact returned pointer using C.free().
// Returns NULL if no match is found.
char* SemanticCache_Get(SemanticCacheHandle cache, const char* doc_id, const float* vector_data, float threshold);

#ifdef __cplusplus
}
#endif

#endif // SEMANTIC_CACHE_H
