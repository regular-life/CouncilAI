package handlers

import (
	"encoding/json"
	"net/http"
	"sync"

	"golang.org/x/crypto/bcrypt"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

// LoginRequest defines credentials for authentication endpoints.
type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// AuthHandler coordinates JWT-based user registrations and logins.
type AuthHandler struct {
	jwtManager *auth.JWTManager
	mu         sync.RWMutex
	users      map[string]string // Simple memory-map for demo. TODO: Persist users in Redis or relational database.
}

// NewAuthHandler initializes AuthHandler with demo credentials.
func NewAuthHandler(jwtManager *auth.JWTManager) *AuthHandler {
	hash, _ := bcrypt.GenerateFromPassword([]byte("demo123"), bcrypt.DefaultCost)
	return &AuthHandler{
		jwtManager: jwtManager,
		users: map[string]string{
			"demo": string(hash),
		},
	}
}

// HandleLogin authenticates users and issues JWT authorization tokens.
func (h *AuthHandler) HandleLogin(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Username == "" || req.Password == "" {
		jsonError(w, "username and password are required", http.StatusBadRequest)
		return
	}

	h.mu.RLock()
	storedHash, exists := h.users[req.Username]
	h.mu.RUnlock()

	if !exists {
		jsonError(w, "invalid credentials", http.StatusUnauthorized)
		return
	}
	if err := bcrypt.CompareHashAndPassword([]byte(storedHash), []byte(req.Password)); err != nil {
		jsonError(w, "invalid credentials", http.StatusUnauthorized)
		return
	}

	token, err := h.jwtManager.GenerateToken(req.Username)
	if err != nil {
		jsonError(w, "failed to generate token", http.StatusInternalServerError)
		return
	}

	jsonResponse(w, map[string]string{
		"token":   token,
		"user_id": req.Username,
	})
}

// HandleRegister creates a new user and issues a JWT token.
func (h *AuthHandler) HandleRegister(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Username == "" || req.Password == "" {
		jsonError(w, "username and password are required", http.StatusBadRequest)
		return
	}

	h.mu.Lock()
	if _, exists := h.users[req.Username]; exists {
		h.mu.Unlock()
		jsonError(w, "user already exists", http.StatusConflict)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		h.mu.Unlock()
		jsonError(w, "failed to hash password", http.StatusInternalServerError)
		return
	}
	h.users[req.Username] = string(hash)
	h.mu.Unlock()

	token, err := h.jwtManager.GenerateToken(req.Username)
	if err != nil {
		jsonError(w, "failed to generate token", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	jsonResponse(w, map[string]string{
		"token":   token,
		"user_id": req.Username,
		"message": "user created successfully",
	})
}
