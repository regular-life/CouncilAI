package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
)

type RedisCache struct {
	client *redis.Client
	ttl    time.Duration
}

func NewRedisCache(addr, password string, db int) *RedisCache {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})
	return &RedisCache{client: client, ttl: 1 * time.Hour}
}

func (c *RedisCache) Ping(ctx context.Context) error {
	return c.client.Ping(ctx).Err()
}

func CacheKey(docID, question string) string {
	hash := sha256.Sum256([]byte(question))
	return fmt.Sprintf("query:%s:%x", docID, hash[:8])
}

func (c *RedisCache) Get(ctx context.Context, key string, dest interface{}) (bool, error) {
	val, err := c.client.Get(ctx, key).Result()
	if err == redis.Nil {
		metrics.CacheHits.WithLabelValues("miss", "l2").Inc()
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("redis get failed: %w", err)
	}

	metrics.CacheHits.WithLabelValues("hit", "l2").Inc()
	if err := json.Unmarshal([]byte(val), dest); err != nil {
		return false, fmt.Errorf("cache unmarshal failed: %w", err)
	}
	return true, nil
}

func (c *RedisCache) Set(ctx context.Context, key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("cache marshal failed: %w", err)
	}
	if err := c.client.Set(ctx, key, string(data), c.ttl).Err(); err != nil {
		return fmt.Errorf("redis set failed: %w", err)
	}
	log.Printf("[Cache] Set key: %s", key)
	return nil
}

func (c *RedisCache) Close() error {
	return c.client.Close()
}
