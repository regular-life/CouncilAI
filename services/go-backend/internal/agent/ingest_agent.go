package agent

import (
	"context"
	"fmt"
	"log"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

// IngestAgent is responsible for tasks during document ingestion, such as summarization.
type IngestAgent struct {
	client llm.LLMClient
}

// NewIngestAgent creates an agent backed by the given LLM client.
func NewIngestAgent(client llm.LLMClient) *IngestAgent {
	return &IngestAgent{client: client}
}

const summarizePrompt = `You are a document summarization agent for a RAG pipeline.
Your job is to read the provided Beginning-Middle-End (BME) sample of a document and generate a concise 3-4 sentence summary of its core topics.
This summary will be used by a routing agent to determine if future user questions actually require retrieving content from this document.
Focus strictly on the factual subjects, entities, and themes present in the text.`

// SummarizeDocument generates a concise summary of the document based on a preview text sample.
func (a *IngestAgent) SummarizeDocument(ctx context.Context, previewText string) (string, error) {
	if previewText == "" {
		return "", fmt.Errorf("preview text is empty")
	}

	userMsg := fmt.Sprintf("Document Sample:\n%s", previewText)

	resp, err := a.client.GenerateChat(ctx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: summarizePrompt},
			{Role: "user", Content: userMsg},
		},
		Temperature: llm.Float64Ptr(0.2),
		MaxTokens:   200,
	})

	if err != nil {
		log.Printf("[IngestAgent] Summarization failed: %v", err)
		return "", err
	}

	return resp.Answer, nil
}
