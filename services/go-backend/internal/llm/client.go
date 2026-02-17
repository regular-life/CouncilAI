package llm

import "context"

type Response struct {
	Answer     string  `json:"answer"`
	Model      string  `json:"model"`
	Confidence float64 `json:"confidence"`
	Usage      Usage   `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type LLMClient interface {
	Generate(ctx context.Context, prompt string) (*Response, error)
	ModelName() string
}
