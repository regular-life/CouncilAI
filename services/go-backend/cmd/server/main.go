package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/api"
	"github.com/regular-life/CouncilAI/go-backend/internal/api/handlers"
	"github.com/regular-life/CouncilAI/go-backend/internal/audit"
	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/cache/fastcache"
	"github.com/regular-life/CouncilAI/go-backend/internal/config"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

func main() {
	log.Println("Starting PadhAI-Dost Go Backend...")

	cfg := config.Load()

	jwtManager := auth.NewJWTManager(cfg.JWTSecret, cfg.JWTExpiration)
	redisCache := cache.NewRedisCache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	semCache := fastcache.NewSemanticCache(5000)
	defer semCache.Destroy()
	auditLogger := audit.NewLogger()

	var councilClients []llm.LLMClient
	for _, model := range cfg.CouncilModels {
		councilClients = append(councilClients, llm.NewOpenRouterClient(
			cfg.OpenRouterAPIKey, cfg.OpenRouterURL, model, cfg.LLMTimeout,
		))
	}

	chairmanClient := llm.NewGeminiClient(cfg.GeminiAPIKey, cfg.ChairmanModel, cfg.StageTimeout)
	councilOrchestrator := council.NewOrchestrator(councilClients, chairmanClient, cfg.StageTimeout)

	h := handlers.NewHandlers(cfg.RAGServiceURL, councilOrchestrator, redisCache, semCache, auditLogger)
	authHandler := handlers.NewAuthHandler(jwtManager)
	router := api.NewRouter(cfg, h, authHandler, jwtManager)

	server := &http.Server{
		Addr:         fmt.Sprintf(":%s", cfg.ServerPort),
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: cfg.RequestTimeout + 5*time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down gracefully...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
		redisCache.Close()
		log.Println("Server stopped")
	}()

	log.Printf("Server listening on :%s", cfg.ServerPort)
	log.Printf("RAG Service URL: %s", cfg.RAGServiceURL)
	log.Printf("Council models: %v", cfg.CouncilModels)
	log.Printf("Chairman model: %s", cfg.ChairmanModel)

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}
