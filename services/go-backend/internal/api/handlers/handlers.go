package handlers

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/regular-life/padhai-dost/go-backend/internal/api/middleware"
	"github.com/regular-life/padhai-dost/go-backend/internal/audit"
	"github.com/regular-life/padhai-dost/go-backend/internal/cache"
	"github.com/regular-life/padhai-dost/go-backend/internal/council"
	"github.com/regular-life/padhai-dost/go-backend/internal/metrics"
)

type Handlers struct {
	RAGServiceURL string
	Council       *council.Orchestrator
	Cache         *cache.RedisCache
	Audit         *audit.Logger
	HTTPClient    *http.Client
}

func NewHandlers(ragURL string, council *council.Orchestrator, redisCache *cache.RedisCache, auditLogger *audit.Logger) *Handlers {
	return &Handlers{
		RAGServiceURL: ragURL,
		Council:       council,
		Cache:         redisCache,
		Audit:         auditLogger,
		HTTPClient:    &http.Client{Timeout: 120 * time.Second},
	}
}

type QueryRequest struct {
	Question string `json:"question"`
	DocID    string `json:"doc_id"`
	TopK     int    `json:"top_k,omitempty"`
}

type QueryResponse struct {
	Answer       string                    `json:"answer"`
	Confidence   float64                   `json:"confidence"`
	Source       string                    `json:"source"`
	Reasoning    string                    `json:"reasoning,omitempty"`
	PeerReviewed bool                      `json:"peer_reviewed"`
	Candidates   []council.CandidateAnswer `json:"candidates,omitempty"`
	Latency      string                    `json:"latency"`
	CacheHit     bool                      `json:"cache_hit"`
}

// HandleQuery checks cache, retrieves document chunks, runs the LLM council,
// caches the result, and returns the synthesized answer.
func (h *Handlers) HandleQuery(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Question == "" {
		jsonError(w, "question is required", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}

	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Question)))[:16]

	cacheKey := cache.CacheKey(req.DocID, req.Question)
	var cachedResponse QueryResponse
	if found, err := h.Cache.Get(r.Context(), cacheKey, &cachedResponse); err == nil && found {
		cachedResponse.CacheHit = true
		cachedResponse.Latency = time.Since(start).String()
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "cache_hit")
		jsonResponse(w, cachedResponse)
		return
	}

	chunks, err := h.retrieveChunks(r, req)
	if err != nil {
		log.Printf("[Query] Retrieval failed: %v", err)
		jsonError(w, "failed to retrieve document chunks", http.StatusInternalServerError)
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "retrieval_error")
		return
	}
	if len(chunks) == 0 {
		jsonError(w, "no relevant chunks found for this question", http.StatusNotFound)
		return
	}

	councilStart := time.Now()
	result, err := h.Council.Query(r.Context(), req.Question, chunks, "", false)
	if err != nil {
		log.Printf("[Query] Council failed: %v", err)
		jsonError(w, "LLM council failed", http.StatusInternalServerError)
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
		return
	}
	metrics.CouncilResponseTime.Observe(time.Since(councilStart).Seconds())

	response := QueryResponse{
		Answer:       result.FinalAnswer,
		Confidence:   result.Confidence,
		Source:       result.Source,
		Reasoning:    result.Reasoning,
		PeerReviewed: len(result.PeerReviews) > 0,
		Candidates:   result.CandidateAnswers,
		Latency:      time.Since(start).String(),
		CacheHit:     false,
	}

	if err := h.Cache.Set(r.Context(), cacheKey, response); err != nil {
		log.Printf("[Query] Cache set failed: %v", err)
	}

	h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	jsonResponse(w, response)
}

// HandleIngest proxies file upload to the Python RAG service.
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
		writer.WriteField("doc_id", docID)
	}
	writer.Close()

	resp, err := h.HTTPClient.Post(h.RAGServiceURL+"/ingest", writer.FormDataContentType(), &buf)
	if err != nil {
		log.Printf("[Ingest] Failed to contact RAG service: %v", err)
		jsonError(w, "RAG service unavailable", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	h.Audit.LogIngest(userID, docID, time.Since(start), "success")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

func (h *Handlers) HandleHealth(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"status":  "healthy",
		"service": "go-backend",
		"version": "1.0.0",
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
		resp.Body.Close()
	}

	jsonResponse(w, status)
}

// --- Explain ---

type ExplainRequest struct {
	DocID          string   `json:"doc_id"`
	KnowledgeLevel string   `json:"knowledge_level"`
	Depth          string   `json:"depth"`
	FocusTopics    []string `json:"focus_topics,omitempty"`
}

type ExplainSection struct {
	Heading  string `json:"heading"`
	Content  string `json:"content"`
	PageRefs []int  `json:"page_refs,omitempty"`
}

type ExplainResponse struct {
	Explanation string           `json:"explanation"`
	Sections    []ExplainSection `json:"sections,omitempty"`
	Confidence  float64          `json:"confidence"`
	Source      string           `json:"source"`
	Latency     string           `json:"latency"`
	CacheHit    bool             `json:"cache_hit"`
}

// HandleExplain generates a knowledge-level-adaptive explanation of a document.
func (h *Handlers) HandleExplain(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	var req ExplainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.DocID == "" {
		jsonError(w, "doc_id is required", http.StatusBadRequest)
		return
	}
	if req.KnowledgeLevel == "" {
		req.KnowledgeLevel = "beginner"
	}
	if req.Depth == "" {
		req.Depth = "section-wise"
	}

	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte("explain:"+req.DocID+":"+req.KnowledgeLevel+":"+req.Depth)))[:16]

	cacheKey := fmt.Sprintf("explain:%s:%s:%s", req.DocID, req.KnowledgeLevel, req.Depth)
	var cachedResponse ExplainResponse
	if found, err := h.Cache.Get(r.Context(), cacheKey, &cachedResponse); err == nil && found {
		cachedResponse.CacheHit = true
		cachedResponse.Latency = time.Since(start).String()
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "cache_hit")
		jsonResponse(w, cachedResponse)
		return
	}

	chunks, err := h.retrieveAllChunks(req.DocID)
	if err != nil {
		log.Printf("[Explain] Retrieval failed: %v", err)
		jsonError(w, "failed to retrieve document", http.StatusInternalServerError)
		return
	}
	if len(chunks) == 0 {
		jsonError(w, "document not found or empty", http.StatusNotFound)
		return
	}

	contextText := strings.Join(chunks, "\n\n---\n\n")

	focusClause := ""
	if len(req.FocusTopics) > 0 {
		focusClause = fmt.Sprintf("\n\nFocus especially on these topics: %s", strings.Join(req.FocusTopics, ", "))
	}

	depthInstruction := map[string]string{
		"brief":        "Provide a concise summary (2-3 paragraphs max).",
		"section-wise": "Organize the explanation into clear sections with headings.",
		"detailed":     "Provide a thorough, detailed explanation covering all major points.",
	}[req.Depth]
	if depthInstruction == "" {
		depthInstruction = "Organize the explanation into clear sections with headings."
	}

	levelInstruction := map[string]string{
		"beginner":     "Explain as if to someone completely new to the subject. Use simple language, analogies, and define all technical terms.",
		"intermediate": "Explain at an intermediate level. Use domain terminology but provide context. Include analysis and comparisons.",
		"advanced":     "Explain at an expert level. Use precise technical language. Include critical analysis, research implications, and connections to related concepts.",
	}[req.KnowledgeLevel]
	if levelInstruction == "" {
		levelInstruction = "Explain as if to someone completely new to the subject."
	}

	prompt := fmt.Sprintf(`You are an expert educator. Based ONLY on the following document excerpts, generate an explanation.

Document Excerpts:
%s

Instructions:
- %s
- %s
- Ground every claim in the provided excerpts — do not invent information
- Use clear formatting with headings and structure%s

Respond with a well-structured explanation.`, contextText, levelInstruction, depthInstruction, focusClause)

	result, err := h.Council.Query(r.Context(), "explain:"+req.DocID, chunks, prompt, false)
	if err != nil {
		log.Printf("[Explain] Council failed: %v", err)
		jsonError(w, "explanation generation failed", http.StatusInternalServerError)
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
		return
	}

	response := ExplainResponse{
		Explanation: result.FinalAnswer,
		Confidence:  result.Confidence,
		Source:      result.Source,
		Latency:     time.Since(start).String(),
		CacheHit:    false,
	}

	if err := h.Cache.Set(r.Context(), cacheKey, response); err != nil {
		log.Printf("[Explain] Cache set failed: %v", err)
	}

	h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	jsonResponse(w, response)
}

// --- Generate Questions ---

type GenerateQuestionsRequest struct {
	DocID        string `json:"doc_id"`
	NumQuestions int    `json:"num_questions"`
	Difficulty   int    `json:"difficulty"`
	QuestionType string `json:"question_type"`
	BloomLevel   string `json:"bloom_level,omitempty"`
}

type GeneratedQuestion struct {
	Question    string   `json:"question"`
	Answer      string   `json:"answer"`
	Explanation string   `json:"explanation"`
	SourceChunk string   `json:"source_chunk,omitempty"`
	Options     []string `json:"options,omitempty"`
}

type GenerateQuestionsResponse struct {
	Questions  []GeneratedQuestion `json:"questions"`
	RawOutput  string              `json:"raw_output,omitempty"`
	Confidence float64             `json:"confidence"`
	Source     string              `json:"source"`
	Latency    string              `json:"latency"`
	CacheHit   bool                `json:"cache_hit"`
}

// HandleGenerateQuestions creates assessment questions from a document.
// Uses skipChairman=true to preserve structured JSON output.
func (h *Handlers) HandleGenerateQuestions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	var req GenerateQuestionsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.DocID == "" {
		jsonError(w, "doc_id is required", http.StatusBadRequest)
		return
	}
	if req.NumQuestions <= 0 {
		req.NumQuestions = 5
	}
	if req.NumQuestions > 20 {
		req.NumQuestions = 20
	}
	if req.Difficulty <= 0 || req.Difficulty > 10 {
		req.Difficulty = 5
	}
	if req.QuestionType == "" {
		req.QuestionType = "subjective"
	}

	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf("questions:%s:%d:%d:%s",
		req.DocID, req.NumQuestions, req.Difficulty, req.QuestionType))))[:16]

	cacheKey := fmt.Sprintf("questions:%s:%d:%d:%s", req.DocID, req.NumQuestions, req.Difficulty, req.QuestionType)
	var cachedResponse GenerateQuestionsResponse
	if found, err := h.Cache.Get(r.Context(), cacheKey, &cachedResponse); err == nil && found {
		cachedResponse.CacheHit = true
		cachedResponse.Latency = time.Since(start).String()
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "cache_hit")
		jsonResponse(w, cachedResponse)
		return
	}

	chunks, err := h.retrieveAllChunks(req.DocID)
	if err != nil {
		log.Printf("[GenerateQuestions] Retrieval failed: %v", err)
		jsonError(w, "failed to retrieve document", http.StatusInternalServerError)
		return
	}
	if len(chunks) == 0 {
		jsonError(w, "document not found or empty", http.StatusNotFound)
		return
	}

	contextText := strings.Join(chunks, "\n\n---\n\n")

	bloomClause := ""
	if req.BloomLevel != "" {
		bloomClause = fmt.Sprintf("\n- Target Bloom's taxonomy level: %s", req.BloomLevel)
	}

	questionTypeInstruction := "open-ended subjective questions requiring evidence-based reasoning"
	if req.QuestionType == "mcq" {
		questionTypeInstruction = "multiple-choice questions (MCQ) with exactly 4 options each (A, B, C, D) and indicate the correct answer"
	}

	prompt := fmt.Sprintf(`You are an expert assessment designer. Based ONLY on the following document excerpts, generate practice questions.

Document Excerpts:
%s

Generate exactly %d %s.

Requirements:
- Difficulty level: %d/10 (1-3: recall, 4-6: analysis/application, 7-10: synthesis/evaluation)%s
- Ground every question in the provided document content
- Each question must be answerable from the excerpts
- Provide an answer and brief explanation for each question
- Do not repeat similar questions

Respond as a JSON array where each element has:
{
  "question": "The question text",
  "answer": "The correct answer",
  "explanation": "Why this is the answer, citing the source material"%s
}

Respond ONLY with the JSON array.`, contextText, req.NumQuestions, questionTypeInstruction, req.Difficulty, bloomClause,
		func() string {
			if req.QuestionType == "mcq" {
				return `,
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."]`
			}
			return ""
		}())

	result, err := h.Council.Query(r.Context(), "questions:"+req.DocID, chunks, prompt, true)
	if err != nil {
		log.Printf("[GenerateQuestions] Council failed: %v", err)
		jsonError(w, "question generation failed", http.StatusInternalServerError)
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
		return
	}

	var questions []GeneratedQuestion
	rawAnswer := result.FinalAnswer

	jsonStr := extractQuestionsJSON(rawAnswer)
	if err := json.Unmarshal([]byte(jsonStr), &questions); err != nil {
		log.Printf("[GenerateQuestions] JSON parse failed, returning raw output: %v", err)
	}
	if len(questions) > req.NumQuestions {
		questions = questions[:req.NumQuestions]
	}

	response := GenerateQuestionsResponse{
		Questions:  questions,
		RawOutput:  rawAnswer,
		Confidence: result.Confidence,
		Source:     result.Source,
		Latency:    time.Since(start).String(),
		CacheHit:   false,
	}

	if err := h.Cache.Set(r.Context(), cacheKey, response); err != nil {
		log.Printf("[GenerateQuestions] Cache set failed: %v", err)
	}

	h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	jsonResponse(w, response)
}

// --- JSON sanitization helpers ---

// extractQuestionsJSON pulls a JSON array out of LLM output that may include
// markdown code fences, control characters, and invalid backslash escapes.
func extractQuestionsJSON(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	}
	start := strings.Index(s, "[")
	end := strings.LastIndex(s, "]")
	if start >= 0 && end > start {
		s = s[start : end+1]
	}
	s = stripControlChars(s)
	s = sanitizeJSONBackslashes(s)
	return s
}

func stripControlChars(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r >= 0x20 || r == '\t' || r == '\n' || r == '\r' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// sanitizeJSONBackslashes double-escapes backslashes that aren't valid JSON
// escape sequences (handles LaTeX like \Lambda, \theta, etc).
func sanitizeJSONBackslashes(s string) string {
	var result strings.Builder
	result.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			next := s[i+1]
			if next == '"' || next == '\\' || next == '/' ||
				next == 'b' || next == 'f' || next == 'n' ||
				next == 'r' || next == 't' || next == 'u' {
				result.WriteByte(s[i])
				result.WriteByte(next)
				i++
			} else {
				result.WriteString("\\\\")
			}
		} else {
			result.WriteByte(s[i])
		}
	}
	return result.String()
}

// --- RAG service communication ---

func (h *Handlers) retrieveAllChunks(docID string) ([]string, error) {
	reqBody := map[string]string{"doc_id": docID}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	resp, err := h.HTTPClient.Post(h.RAGServiceURL+"/retrieve-all", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("retrieve-all request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("retrieve-all failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Chunks []struct {
			Content string `json:"content"`
		} `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	chunks := make([]string, len(result.Chunks))
	for i, c := range result.Chunks {
		chunks[i] = c.Content
	}
	return chunks, nil
}

func (h *Handlers) retrieveChunks(r *http.Request, req QueryRequest) ([]string, error) {
	reqBody := map[string]interface{}{
		"question": req.Question,
		"doc_id":   req.DocID,
		"top_k":    req.TopK,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	resp, err := h.HTTPClient.Post(h.RAGServiceURL+"/retrieve", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("retrieval request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("retrieval failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Chunks []struct {
			Content string `json:"content"`
		} `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	chunks := make([]string, len(result.Chunks))
	for i, c := range result.Chunks {
		chunks[i] = c.Content
	}
	return chunks, nil
}

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func jsonResponse(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}
