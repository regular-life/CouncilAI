package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OpenRouterClient struct {
	apiKey string
	apiURL string
	model  string
	client *http.Client
}

func NewOpenRouterClient(apiKey, apiURL, model string, timeout time.Duration) *OpenRouterClient {
	return &OpenRouterClient{
		apiKey: apiKey,
		apiURL: apiURL,
		model:  model,
		client: &http.Client{Timeout: timeout},
	}
}

type openRouterRequest struct {
	Model    string              `json:"model"`
	Messages []openRouterMessage `json:"messages"`
}

type openRouterMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openRouterResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func (c *OpenRouterClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	reqBody := openRouterRequest{
		Model:    c.model,
		Messages: []openRouterMessage{{Role: "user", Content: prompt}},
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("HTTP-Referer", "https://github.com/regular-life/PadhAI-Dost")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("OpenRouter request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenRouter API error (status %d): %s", resp.StatusCode, string(body))
	}

	var orResp openRouterResponse
	if err := json.Unmarshal(body, &orResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(orResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenRouter response")
	}

	return &Response{
		Answer: orResp.Choices[0].Message.Content,
		Model:  c.model,
		Usage: Usage{
			PromptTokens:     orResp.Usage.PromptTokens,
			CompletionTokens: orResp.Usage.CompletionTokens,
			TotalTokens:      orResp.Usage.TotalTokens,
		},
	}, nil
}

func (c *OpenRouterClient) ModelName() string {
	return c.model
}
