package council

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

type ChairmanResult struct {
	Answer     string  `json:"answer"`
	Reasoning  string  `json:"reasoning"`
	Source     string  `json:"source"`
	Confidence float64 `json:"confidence"`
}

func (o *Orchestrator) chairmanSynthesize(
	question string,
	chunks []string,
	candidates []CandidateAnswer,
	reviews []PeerReview,
) (*ChairmanResult, error) {
	var responsesBlock []string
	for i, c := range candidates {
		responsesBlock = append(responsesBlock, fmt.Sprintf("=== Response %c ===\n%s", 'A'+i, c.Answer))
	}

	var reviewsBlock []string
	for i, r := range reviews {
		if r.Error == "" {
			reviewsBlock = append(reviewsBlock, fmt.Sprintf("--- Reviewer %d ---\n%s", i+1, r.Review))
		}
	}

	reviewsText := "No peer reviews available."
	if len(reviewsBlock) > 0 {
		reviewsText = strings.Join(reviewsBlock, "\n\n")
	}

	prompt := fmt.Sprintf(`You are the Chairman of an AI council. Multiple AI models have answered a question, and their peers have reviewed and ranked the answers. Your job is to synthesize the best possible final answer.

		Question: %s

		Document Excerpts:
		%s

		Individual Responses:
		%s

		Peer Reviews:
		%s

		Your task:
		1. Consider all responses and the peer review feedback
		2. Identify the strongest and weakest points across all answers
		3. Synthesize a single, authoritative final answer
		4. Rate your confidence in the final answer

		Respond in JSON format:
		{
		"answer": "Your synthesized final answer here",
		"reasoning": "Brief explanation of how you synthesized the answer and which responses contributed most",
		"source": "chairman-synthesis",
		"confidence": 0.85
		}

		Respond ONLY with valid JSON, no other text.`,
		question,
		strings.Join(chunks, "\n\n---\n\n"),
		strings.Join(responsesBlock, "\n\n"),
		reviewsText,
	)

	ctx, cancel := context.WithTimeout(context.Background(), o.stageTimeout)
	defer cancel()

	resp, err := o.chairmanClient.Generate(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("chairman generation failed: %w", err)
	}

	result := &ChairmanResult{}
	jsonStr := extractJSON(resp.Answer)
	if err := json.Unmarshal([]byte(jsonStr), result); err != nil {
		log.Printf("[Council] Chairman JSON parse failed, using raw answer: %v", err)
		result = &ChairmanResult{
			Answer:     resp.Answer,
			Reasoning:  "Chairman response was not valid JSON",
			Source:     "chairman:" + o.chairmanClient.ModelName(),
			Confidence: 0.7,
		}
	}
	if result.Source == "" {
		result.Source = "chairman:" + o.chairmanClient.ModelName()
	}

	return result, nil
}

// extractJSON pulls a JSON object out of LLM output that may include markdown
// code fences, control characters, and invalid backslash escapes.
func extractJSON(s string) string {
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

	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		s = s[start : end+1]
	}

	// Strip control characters
	var cleaned strings.Builder
	cleaned.Grow(len(s))
	for _, r := range s {
		if r >= 0x20 || r == '\t' || r == '\n' || r == '\r' {
			cleaned.WriteRune(r)
		}
	}
	s = cleaned.String()

	s = sanitizeJSONBackslashes(s)
	return s
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
