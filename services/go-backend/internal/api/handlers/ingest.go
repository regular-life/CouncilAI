package handlers

import (
	"bytes"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
)

// HandleIngest passes uploaded documents to the Python RAG service.
// TODO: Support multi-part file uploads of other formats (docx, txt).
func (h *Handlers) HandleIngest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	if err := r.ParseMultipartForm(50 << 20); err != nil {
		jsonError(w, "failed to parse form data", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		jsonError(w, "file is required", http.StatusBadRequest)
		return
	}
	defer file.Close()

	docID := r.FormValue("doc_id")

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", header.Filename)
	if err != nil {
		jsonError(w, "failed to create form file", http.StatusInternalServerError)
		return
	}
	if _, err := io.Copy(part, file); err != nil {
		jsonError(w, "failed to copy file", http.StatusInternalServerError)
		return
	}
	if docID != "" {
		_ = writer.WriteField("doc_id", docID)
	}
	writer.Close()

	resp, err := h.HTTPClient.Post(h.RAGServiceURL+"/ingest", writer.FormDataContentType(), &buf)
	if err != nil {
		log.Printf("[Ingest] Failed to contact RAG service: %v", err)
		jsonError(w, "RAG service unavailable", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Ingest] Failed to read response body: %v", err)
		jsonError(w, "failed to read ingestion response", http.StatusInternalServerError)
		return
	}
	h.Audit.LogIngest(userID, docID, time.Since(start), "success")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}

// HandleHealth checks Go control plane health and dependency states.
// TODO: Return detailed status report on pipeline component health.
func (h *Handlers) HandleHealth(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"status":  "healthy",
		"service": "go-backend",
		"version": "2.0.0",
	}

	if err := h.Cache.Ping(r.Context()); err != nil {
		status["redis"] = "unhealthy"
	} else {
		status["redis"] = "healthy"
	}

	resp, err := h.HTTPClient.Get(h.RAGServiceURL + "/health")
	if err != nil || resp.StatusCode != 200 {
		status["rag_service"] = "unhealthy"
	} else {
		status["rag_service"] = "healthy"
		_ = resp.Body.Close()
	}

	jsonResponse(w, status)
}
