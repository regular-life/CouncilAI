package council

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
)

type CouncilResult struct {
	FinalAnswer      string            `json:"final_answer"`
	Reasoning        string            `json:"reasoning,omitempty"`
	Confidence       float64           `json:"confidence"`
	Source           string            `json:"source"`
	CandidateAnswers []CandidateAnswer `json:"candidate_answers"`
	PeerReviews      []PeerReview      `json:"peer_reviews,omitempty"`
	Latency          time.Duration     `json:"latency"`
}

type CandidateAnswer struct {
	Answer string `json:"answer"`
	Model  string `json:"model"`
	Error  string `json:"error,omitempty"`
}

type PeerReview struct {
	Reviewer string `json:"reviewer"`
	Review   string `json:"review"`
	Error    string `json:"error,omitempty"`
}

type Orchestrator struct {
	clients        []llm.LLMClient
	chairmanClient llm.LLMClient
	stageTimeout   time.Duration
}

func NewOrchestrator(clients []llm.LLMClient, chairmanClient llm.LLMClient, stageTimeout time.Duration) *Orchestrator {
	if stageTimeout == 0 {
		stageTimeout = 30 * time.Second
	}
	return &Orchestrator{
		clients:        clients,
		chairmanClient: chairmanClient,
		stageTimeout:   stageTimeout,
	}
}

func (o *Orchestrator) Query(ctx context.Context, question string, chunks []string, customPrompt string, skipChairman bool) (*CouncilResult, error) {
	start := time.Now()

	if os.Getenv("MOCK_LLM") == "true" {
		time.Sleep(300 * time.Millisecond) // Simulate a very fast LLM
		return &CouncilResult{
			FinalAnswer: "MOCK RESPONSE: Answer generated locally to preserve API quotas. Original Question: " + question,
			Confidence:  0.99,
			Source:      "mock:council",
			Latency:     time.Since(start),
		}, nil
	}

	prompt := customPrompt
	if prompt == "" {
		contextText := strings.Join(chunks, "\n\n---\n\n")
		prompt = fmt.Sprintf(`Based on the following document excerpts, answer the question.

		Document Excerpts:
		%s

		Question: %s

		Instructions:
		- Answer based ONLY on the provided excerpts
		- If the answer is not in the excerpts, say so clearly
		- Be concise but thorough`, contextText, question)
	}

	log.Printf("[Council] Collecting individual responses from %d models", len(o.clients))
	candidates := o.fanOut(prompt)

	var valid []CandidateAnswer
	for _, c := range candidates {
		if c.Error == "" {
			valid = append(valid, c)
		}
	}
	if len(valid) == 0 {
		return nil, fmt.Errorf("all council members failed to respond")
	}
	if len(valid) == 1 {
		log.Printf("[Council] Only 1 model responded, skipping peer review and chairman")
		return &CouncilResult{
			FinalAnswer:      valid[0].Answer,
			Confidence:       0.5,
			Source:           valid[0].Model + " (single-response)",
			CandidateAnswers: candidates,
			Latency:          time.Since(start),
		}, nil
	}

	log.Printf("[Council] Peer review with %d valid candidates", len(valid))
	reviews := o.peerReview(question, valid)

	successfulReviews := 0
	for _, r := range reviews {
		if r.Error == "" {
			successfulReviews++
		}
	}
	if successfulReviews == 0 {
		log.Printf("[Council] Peer review failed entirely, falling back to best candidate")
		best := pickLongestCandidate(valid)
		return &CouncilResult{
			FinalAnswer:      best.Answer,
			Confidence:       0.6,
			Source:           best.Model + " (peer-review-failed-fallback)",
			CandidateAnswers: candidates,
			PeerReviews:      reviews,
			Latency:          time.Since(start),
		}, nil
	}

	if skipChairman {
		log.Printf("[Council] Skipping chairman (skipChairman=true), using best peer-reviewed candidate")
		best := pickLongestCandidate(valid)
		metrics.CouncilResponseTime.Observe(time.Since(start).Seconds())
		return &CouncilResult{
			FinalAnswer:      best.Answer,
			Confidence:       0.75,
			Source:           best.Model + " (peer-reviewed, no chairman)",
			CandidateAnswers: candidates,
			PeerReviews:      reviews,
			Latency:          time.Since(start),
		}, nil
	}

	log.Printf("[Council] Chairman synthesis")
	chairmanResult, err := o.chairmanSynthesize(question, chunks, valid, reviews)
	if err != nil {
		log.Printf("[Council] Chairman synthesis failed: %v, falling back to best candidate", err)
		best := pickLongestCandidate(valid)
		return &CouncilResult{
			FinalAnswer:      best.Answer,
			Confidence:       0.65,
			Source:           best.Model + " (chairman-failed-fallback)",
			CandidateAnswers: candidates,
			PeerReviews:      reviews,
			Latency:          time.Since(start),
		}, nil
	}

	metrics.CouncilResponseTime.Observe(time.Since(start).Seconds())
	return &CouncilResult{
		FinalAnswer:      chairmanResult.Answer,
		Reasoning:        chairmanResult.Reasoning,
		Confidence:       chairmanResult.Confidence,
		Source:           chairmanResult.Source,
		CandidateAnswers: candidates,
		PeerReviews:      reviews,
		Latency:          time.Since(start),
	}, nil
}

func (o *Orchestrator) fanOut(prompt string) []CandidateAnswer {
	var wg sync.WaitGroup
	results := make([]CandidateAnswer, len(o.clients))

	for i, client := range o.clients {
		wg.Add(1)
		go func(idx int, c llm.LLMClient) {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), o.stageTimeout)
			defer cancel()

			resp, err := c.Generate(ctx, prompt)
			if err != nil {
				log.Printf("[Council] Model %s failed: %v", c.ModelName(), err)
				metrics.LLMFailureCount.Inc()
				results[idx] = CandidateAnswer{Model: c.ModelName(), Error: err.Error()}
				return
			}
			results[idx] = CandidateAnswer{Answer: resp.Answer, Model: resp.Model}
		}(i, client)
	}

	wg.Wait()
	return results
}

func (o *Orchestrator) peerReview(question string, candidates []CandidateAnswer) []PeerReview {
	var wg sync.WaitGroup
	reviews := make([]PeerReview, len(o.clients))

	var anonymized []string
	for i, c := range candidates {
		anonymized = append(anonymized, fmt.Sprintf("=== Response %c ===\n%s", 'A'+i, c.Answer))
	}
	answersBlock := strings.Join(anonymized, "\n\n")

	for i, client := range o.clients {
		wg.Add(1)
		go func(idx int, c llm.LLMClient) {
			defer wg.Done()

			prompt := fmt.Sprintf(`You are reviewing multiple AI-generated answers to this question:

Question: %s

Here are the anonymized responses:

%s

Your task:
1. Evaluate each response for accuracy, completeness, and clarity
2. Rank them from best to worst
3. Explain briefly why you ranked them that way

Format your response as:
RANKING: [best to worst, e.g., "B, A, C"]
REASONING: [1-2 sentence explanation of your ranking]`, question, answersBlock)

			ctx, cancel := context.WithTimeout(context.Background(), o.stageTimeout)
			defer cancel()

			resp, err := c.Generate(ctx, prompt)
			if err != nil {
				log.Printf("[Council] Reviewer %s failed: %v", c.ModelName(), err)
				reviews[idx] = PeerReview{Reviewer: c.ModelName(), Error: err.Error()}
				return
			}
			reviews[idx] = PeerReview{Reviewer: c.ModelName(), Review: resp.Answer}
		}(i, client)
	}

	wg.Wait()
	return reviews
}

func pickLongestCandidate(candidates []CandidateAnswer) CandidateAnswer {
	best := candidates[0]
	for _, c := range candidates[1:] {
		if len(c.Answer) > len(best.Answer) {
			best = c
		}
	}
	return best
}
