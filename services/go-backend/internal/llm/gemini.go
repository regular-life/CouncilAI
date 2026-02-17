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

type GeminiClient struct {
	apiKey string
	model  string
	client *http.Client
}

func NewGeminiClient(apiKey, model string, timeout time.Duration) *GeminiClient {
	return &GeminiClient{
		apiKey: apiKey,
		model:  model,
		client: &http.Client{Timeout: timeout},
	}
}

type geminiRequest struct {
	Contents []geminiContent `json:"contents"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func (c *GeminiClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	reqBody := geminiRequest{
		Contents: []geminiContent{{
			Parts: []geminiPart{{Text: prompt}},
		}},
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		c.model, c.apiKey,
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Gemini request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Gemini API error (status %d): %s", resp.StatusCode, string(body))
	}

	var gResp geminiResponse
	if err := json.Unmarshal(body, &gResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(gResp.Candidates) == 0 || len(gResp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content in Gemini response")
	}

	return &Response{
		Answer: gResp.Candidates[0].Content.Parts[0].Text,
		Model:  c.model,
		Usage: Usage{
			PromptTokens:     gResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: gResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      gResp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}

func (c *GeminiClient) ModelName() string {
	return "gemini:" + c.model
}
