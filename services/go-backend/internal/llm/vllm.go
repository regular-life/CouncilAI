package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// VLLMClient talks to a local vLLM server via the OpenAI-compatible API.
// No API key is needed since vLLM runs as a local service.
type VLLMClient struct {
	apiURL string // full chat completions URL, e.g. http://vllm-inference:8001/v1/chat/completions
	model  string
	client *http.Client
}

// NewVLLMClient creates a client for a local vLLM inference server.
// apiURL should be the full chat completions endpoint
// (e.g. "http://vllm-inference:8001/v1/chat/completions").
func NewVLLMClient(apiURL, model string, timeout time.Duration) *VLLMClient {
	return &VLLMClient{
		apiURL: apiURL,
		model:  model,
		client: &http.Client{Timeout: timeout},
	}
}

// ── Interface implementation ────────────────────────────────────────

// Generate sends a single prompt — backward-compatible wrapper around GenerateChat.
func (c *VLLMClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{
		Messages: []Message{{Role: "user", Content: prompt}},
	})
}

// GenerateChat sends a full multi-turn conversation to the vLLM server.
func (c *VLLMClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	reqBody := openAIRequest{
		Model: c.model,
	}

	for _, msg := range opts.Messages {
		reqBody.Messages = append(reqBody.Messages, openAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	if opts.Temperature != nil {
		reqBody.Temperature = opts.Temperature
	}
	if opts.MaxTokens > 0 {
		reqBody.MaxTokens = opts.MaxTokens
	}
	if opts.ResponseJSON {
		reqBody.ResponseFormat = &openAIRespFormat{Type: "json_object"}
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal vLLM request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create vLLM request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// No Authorization header — vLLM is a local service.

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vLLM request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read vLLM response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vLLM API error (status %d): %s", resp.StatusCode, string(body))
	}

	var vResp openAIResponse
	if err := json.Unmarshal(body, &vResp); err != nil {
		return nil, fmt.Errorf("failed to parse vLLM response: %w", err)
	}

	if len(vResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in vLLM response")
	}

	return &Response{
		Answer: vResp.Choices[0].Message.Content,
		Model:  c.model,
		Usage: Usage{
			PromptTokens:     vResp.Usage.PromptTokens,
			CompletionTokens: vResp.Usage.CompletionTokens,
			TotalTokens:      vResp.Usage.TotalTokens,
		},
	}, nil
}

// ModelName returns the model identifier prefixed with "local:".
func (c *VLLMClient) ModelName() string {
	return "local:" + c.model
}

// IsReady checks whether the vLLM server is up by hitting its /health endpoint.
func (c *VLLMClient) IsReady(ctx context.Context) bool {
	// Derive base URL by trimming the chat completions path.
	baseURL := strings.TrimSuffix(c.apiURL, "/v1/chat/completions")
	healthURL := baseURL + "/health"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return false
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}
