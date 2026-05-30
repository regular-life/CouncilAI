package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
)

// HandleClearConversation flushes multi-turn chat history for a session.
// TODO: Support selective purging of individual turns or scaling limits by message ID.
func (h *Handlers) HandleClearConversation(w http.ResponseWriter, r *http.Request) {
	userID := middleware.GetUserID(r.Context())

	var req struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.SessionID == "" {
		jsonError(w, "session_id is required", http.StatusBadRequest)
		return
	}

	if h.Memory == nil {
		jsonError(w, "conversation memory not enabled", http.StatusServiceUnavailable)
		return
	}

	if err := h.Memory.Clear(r.Context(), userID, req.SessionID); err != nil {
		log.Printf("[Conversation] Clear failed: %v", err)
		jsonError(w, "failed to clear conversation", http.StatusInternalServerError)
		return
	}

	jsonResponse(w, map[string]string{"status": "cleared"})
}
