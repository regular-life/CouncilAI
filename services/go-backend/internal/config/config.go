package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	ServerPort string
	Debug      bool

	RAGServiceURL string

	RedisAddr     string
	RedisPassword string
	RedisDB       int

	JWTSecret     string
	JWTExpiration time.Duration

	RateLimitRPS   int
	RateLimitBurst int

	OpenRouterAPIKey string
	OpenRouterURL    string
	GeminiAPIKey     string

	CouncilModels  []string
	ChairmanModel  string

	StageTimeout   time.Duration
	LLMTimeout     time.Duration
	RequestTimeout time.Duration
}

func Load() *Config {
	return &Config{
		ServerPort: getEnv("SERVER_PORT", "8080"),
		Debug:      getEnvBool("DEBUG", false),

		RAGServiceURL: getEnv("RAG_SERVICE_URL", "http://python-rag:8000"),

		RedisAddr:     getEnv("REDIS_ADDR", "redis:6379"),
		RedisPassword: getEnv("REDIS_PASSWORD", ""),
		RedisDB:       getEnvInt("REDIS_DB", 0),

		JWTSecret:     getEnv("JWT_SECRET", "padhai-dost-secret-change-me"),
		JWTExpiration: getEnvDuration("JWT_EXPIRATION", 24*time.Hour),

		RateLimitRPS:   getEnvInt("RATE_LIMIT_RPS", 10),
		RateLimitBurst: getEnvInt("RATE_LIMIT_BURST", 20),

		OpenRouterAPIKey: getEnv("OPENROUTER_API_KEY", ""),
		OpenRouterURL:    getEnv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"),
		GeminiAPIKey:     getEnv("GEMINI_API_KEY", ""),

		CouncilModels: []string{
			getEnv("COUNCIL_MODEL_1", "arcee-ai/trinity-large-preview:free"),
			getEnv("COUNCIL_MODEL_2", "stepfun/step-3.5-flash:free"),
			getEnv("COUNCIL_MODEL_3", "nvidia/nemotron-3-nano-30b-a3b:free"),
		},
		ChairmanModel: getEnv("CHAIRMAN_MODEL", "gemini-3-flash-preview"),

		StageTimeout:   getEnvDuration("STAGE_TIMEOUT", 30*time.Second),
		LLMTimeout:     getEnvDuration("LLM_TIMEOUT", 120*time.Second),
		RequestTimeout: getEnvDuration("REQUEST_TIMEOUT", 120*time.Second),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func getEnvDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}
