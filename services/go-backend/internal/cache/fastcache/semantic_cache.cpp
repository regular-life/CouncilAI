#include "semantic_cache.h"
#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// The bge-small-en-v1.5 model returns 384-dimensional vectors.
constexpr int kDim = 384;

struct CacheEntry {
    std::vector<float> vector_data;
    std::string answer;
    std::string doc_id;
};

class SemanticCache {
public:
    explicit SemanticCache(int capacity) : capacity_(capacity) {}

    void Put(const std::string& doc_id, const float* vector_data, const std::string& answer) {
        std::unique_lock<std::shared_mutex> lock(rw_lock_);

        lru_list_.push_front({std::vector<float>(vector_data, vector_data + kDim), answer, doc_id});
        doc_map_[doc_id].push_back(lru_list_.begin());

        if (lru_list_.size() > capacity_) {
            auto last = std::prev(lru_list_.end());
            std::string old_doc = last->doc_id;
            
            auto& entries = doc_map_[old_doc];
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (*it == last) {
                    entries.erase(it);
                    break;
                }
            }
            if (entries.empty()) {
                doc_map_.erase(old_doc);
            }
            lru_list_.pop_back();
        }
    }

    char* Get(const std::string& doc_id, const float* query_vector, float threshold) {
        std::shared_lock<std::shared_mutex> lock(rw_lock_);

        auto it = doc_map_.find(doc_id);
        if (it == doc_map_.end() || it->second.empty()) return nullptr;

        float best_sim = -1.0f;
        std::string best_answer;

        for (auto entry_it : it->second) {
            float sim = CosineSimilaritySimd(query_vector, entry_it->vector_data.data());
            if (sim > best_sim) {
                best_sim = sim;
                best_answer = entry_it->answer;
            }
        }

        if (best_sim >= threshold) {
            return strdup(best_answer.c_str());
        }

        return nullptr;
    }

private:
    float CosineSimilaritySimd(const float* a, const float* b) const {
        __m256 sum_ab = _mm256_setzero_ps();
        __m256 sum_a2 = _mm256_setzero_ps();
        __m256 sum_b2 = _mm256_setzero_ps();

        for (int i = 0; i < kDim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);

            sum_ab = _mm256_add_ps(_mm256_mul_ps(va, vb), sum_ab);
            sum_a2 = _mm256_add_ps(_mm256_mul_ps(va, va), sum_a2);
            sum_b2 = _mm256_add_ps(_mm256_mul_ps(vb, vb), sum_b2);
        }

        float ab[8], a2[8], b2[8];
        _mm256_storeu_ps(ab, sum_ab);
        _mm256_storeu_ps(a2, sum_a2);
        _mm256_storeu_ps(b2, sum_b2);

        float dot = 0, norm_a = 0, norm_b = 0;
        for (int i = 0; i < 8; ++i) {
            dot += ab[i];
            norm_a += a2[i];
            norm_b += b2[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    int capacity_;
    std::list<CacheEntry> lru_list_;
    std::unordered_map<std::string, std::vector<std::list<CacheEntry>::iterator>> doc_map_;
    std::shared_mutex rw_lock_;
};

extern "C" {

SemanticCacheHandle SemanticCache_Create(int capacity) {
    return new SemanticCache(capacity);
}

void SemanticCache_Destroy(SemanticCacheHandle cache) {
    if (cache) {
        delete static_cast<SemanticCache*>(cache);
    }
}

void SemanticCache_Put(SemanticCacheHandle cache, const char* doc_id, const float* vector_data, const char* answer) {
    if (!cache) return;
    static_cast<SemanticCache*>(cache)->Put(doc_id, vector_data, answer);
}

char* SemanticCache_Get(SemanticCacheHandle cache, const char* doc_id, const float* vector_data, float threshold) {
    if (!cache) return nullptr;
    return static_cast<SemanticCache*>(cache)->Get(doc_id, vector_data, threshold);
}

}
