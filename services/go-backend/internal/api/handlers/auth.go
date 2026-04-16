package handlers

import (
	"encoding/json"
	"net/http"

	"golang.org/x/crypto/bcrypt"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

type AuthHandler struct {
	jwtManager *auth.JWTManager
	users      map[string]string
}

func NewAuthHandler(jwtManager *auth.JWTManager) *AuthHandler {
	hash, _ := bcrypt.GenerateFromPassword([]byte("demo123"), bcrypt.DefaultCost)
	return &AuthHandler{
		jwtManager: jwtManager,
		users: map[string]string{
			"demo": string(hash),
		},
	}
}

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

	storedHash, exists := h.users[req.Username]
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
	if _, exists := h.users[req.Username]; exists {
		jsonError(w, "user already exists", http.StatusConflict)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		jsonError(w, "failed to hash password", http.StatusInternalServerError)
		return
	}
	h.users[req.Username] = string(hash)

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
