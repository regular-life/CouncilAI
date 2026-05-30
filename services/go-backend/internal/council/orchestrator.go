package council

import (
	"context"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
)

// CouncilResult holds the full output from a council deliberation.
type CouncilResult struct {
	FinalAnswer      string            `json:"final_answer"`
	Reasoning        string            `json:"reasoning,omitempty"`
	Confidence       float64           `json:"confidence"`
	Source           string            `json:"source"`
	Strategy         string            `json:"strategy,omitempty"`
	CandidateAnswers []CandidateAnswer `json:"candidate_answers"`
	PeerReviews      []PeerReview      `json:"peer_reviews,omitempty"`
	Reflection       *ReflectionResult `json:"reflection,omitempty"`
	Latency          time.Duration     `json:"latency"`
}

// CandidateAnswer is a single model's response to the query.
type CandidateAnswer struct {
	Answer string `json:"answer"`
	Model  string `json:"model"`
	Error  string `json:"error,omitempty"`
}

// PeerReview is a single model's evaluation of the other models' responses.
type PeerReview struct {
	Reviewer string `json:"reviewer"`
	Review   string `json:"review"`
	Error    string `json:"error,omitempty"`
}

// ConversationTurn represents a single turn in a conversation for multi-turn context.
type ConversationTurn struct {
	Role    string `json:"role"`    // "user" | "assistant"
	Content string `json:"content"`
}

// Orchestrator coordinates the multi-model council deliberation.
type Orchestrator struct {
	clients        []llm.LLMClient
	chairmanClient llm.LLMClient
	stageTimeout   time.Duration
}

// NewOrchestrator creates a council orchestrator with N council members and a chairman.
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

// QueryDirect handles the "direct" strategy — single model, no council.
// Used for simple questions that don't need multi-model deliberation.
func (o *Orchestrator) QueryDirect(ctx context.Context, question string, chunks []string, history []ConversationTurn) (*CouncilResult, error) {
	start := time.Now()

	if os.Getenv("MOCK_LLM") == "true" {
		time.Sleep(100 * time.Millisecond)
		return &CouncilResult{
			FinalAnswer: "MOCK RESPONSE (direct): " + question,
			Confidence:  0.99,
			Source:      "mock:direct",
			Strategy:    "direct",
			Latency:     time.Since(start),
		}, nil
	}

	messages := buildMessages(question, chunks, history, "direct")

	directCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
	defer cancel()

	enableSearch := len(chunks) == 0 && strings.HasPrefix(o.chairmanClient.ModelName(), "gemini:")
	resp, err := o.chairmanClient.GenerateChat(directCtx, llm.GenerateOptions{
		Messages:     messages,
		EnableSearch: enableSearch,
	})
	if err != nil {
		return nil, fmt.Errorf("direct query failed: %w", err)
	}

	return &CouncilResult{
		FinalAnswer: resp.Answer,
		Confidence:  0.8,
		Source:      "direct:" + o.chairmanClient.ModelName(),
		Strategy:    "direct",
		Latency:     time.Since(start),
	}, nil
}

// Query runs the full council deliberation pipeline.
// strategy should be "council" or "council_deep".
// chunks can be empty (general knowledge mode).
// history provides multi-turn conversation context.
func (o *Orchestrator) Query(ctx context.Context, question string, chunks []string, customPrompt string, skipChairman bool, strategy string, history []ConversationTurn) (*CouncilResult, error) {
	start := time.Now()

	if strategy == "" {
		strategy = "council"
	}

	if os.Getenv("MOCK_LLM") == "true" {
		time.Sleep(300 * time.Millisecond)
		return &CouncilResult{
			FinalAnswer: "MOCK RESPONSE: Answer generated locally to preserve API quotas. Original Question: " + question,
			Confidence:  0.99,
			Source:      "mock:council",
			Strategy:    strategy,
			Latency:     time.Since(start),
		}, nil
	}

	// Build the prompt for council members
	prompt := customPrompt
	if prompt == "" {
		messages := buildMessages(question, chunks, history, "council")
		// For backward compatibility with fan-out (which uses Generate),
		// flatten messages into a single prompt string
		prompt = flattenMessages(messages)
	}

	// ── Stage 1: Fan-out to all council members ──────────────────────
	log.Printf("[Council] Collecting individual responses from %d models", len(o.clients))
	candidates := o.fanOut(ctx, prompt)

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
			Strategy:         strategy,
			CandidateAnswers: candidates,
			Latency:          time.Since(start),
		}, nil
	}

	// ── Stage 2: Peer review ─────────────────────────────────────────
	log.Printf("[Council] Peer review with %d valid candidates", len(valid))
	reviews := o.peerReview(ctx, question, valid)

	successfulReviews := 0
	for _, r := range reviews {
		if r.Error == "" {
			successfulReviews++
		}
	}
	if successfulReviews == 0 {
		log.Printf("[Council] Peer review failed entirely, falling back to best candidate")
		best := pickBestCandidate(valid, nil)
		return &CouncilResult{
			FinalAnswer:      best.Answer,
			Confidence:       0.6,
			Source:           best.Model + " (peer-review-failed-fallback)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			PeerReviews:      reviews,
			Latency:          time.Since(start),
		}, nil
	}

	if skipChairman {
		log.Printf("[Council] Skipping chairman (skipChairman=true), using best peer-reviewed candidate")
		best := pickBestCandidate(valid, reviews)
		metrics.CouncilResponseTime.Observe(time.Since(start).Seconds())
		return &CouncilResult{
			FinalAnswer:      best.Answer,
			Confidence:       0.75,
			Source:           best.Model + " (peer-reviewed, no chairman)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			PeerReviews:      reviews,
			Latency:          time.Since(start),
		}, nil
	}

	// ── Stage 3: Chairman synthesis ──────────────────────────────────
	log.Printf("[Council] Chairman synthesis")
	chairmanResult, err := o.chairmanSynthesize(question, chunks, valid, reviews)
	if err != nil {
		log.Printf("[Council] Chairman synthesis failed: %v, falling back to best candidate", err)
		best := pickBestCandidate(valid, reviews)
		return &CouncilResult{
			FinalAnswer:      best.Answer,
			Confidence:       0.65,
			Source:           best.Model + " (chairman-failed-fallback)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			PeerReviews:      reviews,
			Latency:          time.Since(start),
		}, nil
	}

	result := &CouncilResult{
		FinalAnswer:      chairmanResult.Answer,
		Reasoning:        chairmanResult.Reasoning,
		Confidence:       chairmanResult.Confidence,
		Source:           chairmanResult.Source,
		Strategy:         strategy,
		CandidateAnswers: candidates,
		PeerReviews:      reviews,
		Latency:          time.Since(start),
	}

	// ── Stage 4 (optional): Reflection — only for council_deep ──────
	if strategy == "council_deep" {
		log.Printf("[Council] Running reflection loop (council_deep strategy)")
		reflection, err := o.Reflect(ctx, question, chunks, chairmanResult.Answer)
		if err != nil {
			log.Printf("[Council] Reflection failed: %v, keeping original answer", err)
		} else {
			result.Reflection = reflection
			log.Printf("[Council] Reflection: quality=%s faithful=%v confidence=%.2f",
				reflection.Quality, reflection.Faithful, reflection.Confidence)

			if reflection.Quality == "needs_revision" {
				log.Printf("[Council] Answer needs revision, running one more chairman pass")
				revised, err := o.reviseAnswer(ctx, question, chunks, chairmanResult.Answer, reflection)
				if err != nil {
					log.Printf("[Council] Revision failed: %v, keeping original answer", err)
				} else {
					result.FinalAnswer = revised.Answer
					result.Reasoning = revised.Reasoning
					result.Confidence = revised.Confidence
					result.Source = revised.Source
				}
			} else {
				// Update confidence from reflection if it's available
				if reflection.Confidence > 0 {
					result.Confidence = reflection.Confidence
				}
			}
		}
	}

	metrics.CouncilResponseTime.Observe(time.Since(start).Seconds())
	result.Latency = time.Since(start)
	return result, nil
}

// ── Stage implementations ───────────────────────────────────────────

func (o *Orchestrator) fanOut(ctx context.Context, prompt string) []CandidateAnswer {
	var wg sync.WaitGroup
	results := make([]CandidateAnswer, len(o.clients))

	for i, client := range o.clients {
		wg.Add(1)
		go func(idx int, c llm.LLMClient) {
			defer wg.Done()

			subCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
			defer cancel()

			enableSearch := strings.HasPrefix(c.ModelName(), "gemini:") && !strings.Contains(prompt, "Document Excerpts:")
			resp, err := c.GenerateChat(subCtx, llm.GenerateOptions{
				Messages:     []llm.Message{{Role: "user", Content: prompt}},
				EnableSearch: enableSearch,
			})
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

func (o *Orchestrator) peerReview(ctx context.Context, question string, candidates []CandidateAnswer) []PeerReview {
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

			subCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
			defer cancel()

			resp, err := c.Generate(subCtx, prompt)
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

// ── Candidate selection (replaces pickLongestCandidate) ─────────────

// rankingPattern matches "RANKING: B, A, C" or "RANKING: [B, A, C]"
var rankingPattern = regexp.MustCompile(`(?i)RANKING:\s*\[?\s*([A-Z](?:\s*,\s*[A-Z])*)\s*\]?`)

// pickBestCandidate selects the best candidate based on peer review rankings.
// Falls back to longest candidate if rankings can't be parsed.
func pickBestCandidate(candidates []CandidateAnswer, reviews []PeerReview) CandidateAnswer {
	if len(candidates) == 0 {
		return CandidateAnswer{}
	}

	if len(reviews) == 0 {
		return pickLongest(candidates)
	}

	// Score each candidate based on peer review rankings
	scores := make(map[int]int) // candidate index → aggregate score
	parsedAny := false

	for _, review := range reviews {
		if review.Error != "" {
			continue
		}

		matches := rankingPattern.FindStringSubmatch(review.Review)
		if len(matches) < 2 {
			continue
		}

		// Parse "B, A, C" → ['B', 'A', 'C']
		rankStr := strings.ReplaceAll(matches[1], " ", "")
		letters := strings.Split(rankStr, ",")

		n := len(candidates)
		for rank, letter := range letters {
			if len(letter) != 1 {
				continue
			}
			idx := int(strings.ToUpper(letter)[0] - 'A')
			if idx >= 0 && idx < n {
				// Points: 1st place gets N points, 2nd gets N-1, etc.
				scores[idx] += n - rank
				parsedAny = true
			}
		}
	}

	if !parsedAny {
		log.Printf("[Council] Could not parse any peer review rankings, falling back to longest")
		return pickLongest(candidates)
	}

	// Find candidate with highest score
	bestIdx := 0
	bestScore := -1
	for idx, score := range scores {
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}

	log.Printf("[Council] Peer review scores: %v → selected %c (%s)", scores, 'A'+bestIdx, candidates[bestIdx].Model)
	return candidates[bestIdx]
}

// pickLongest selects the candidate with the longest answer (fallback).
func pickLongest(candidates []CandidateAnswer) CandidateAnswer {
	best := candidates[0]
	for _, c := range candidates[1:] {
		if len(c.Answer) > len(best.Answer) {
			best = c
		}
	}
	return best
}

// ── Prompt building ─────────────────────────────────────────────────

// buildMessages constructs the LLM message list based on context.
func buildMessages(question string, chunks []string, history []ConversationTurn, strategy string) []llm.Message {
	var messages []llm.Message

	// System prompt — adapts based on whether documents are present
	if len(chunks) > 0 {
		messages = append(messages, llm.Message{
			Role: "system",
			Content: `You are a knowledgeable AI assistant. Answer the question based on the provided document excerpts. 
If the answer is not in the excerpts, say so clearly. Be concise but thorough. Cite relevant parts of the source material when possible.`,
		})
	} else {
		messages = append(messages, llm.Message{
			Role: "system",
			Content: `You are a knowledgeable AI assistant. Answer the question accurately and thoroughly using your knowledge. 
Be honest when uncertain. Provide reasoning and cite your sources of knowledge when relevant.`,
		})
	}

	// Conversation history (multi-turn context)
	for _, turn := range history {
		messages = append(messages, llm.Message{
			Role:    turn.Role,
			Content: turn.Content,
		})
	}

	// Build the user message with optional document context
	userContent := ""
	if len(chunks) > 0 {
		contextText := strings.Join(chunks, "\n\n---\n\n")
		userContent = fmt.Sprintf("Document Excerpts:\n%s\n\nQuestion: %s", contextText, question)
	} else {
		userContent = question
	}

	messages = append(messages, llm.Message{
		Role:    "user",
		Content: userContent,
	})

	return messages
}

// flattenMessages converts a message list into a single prompt string
// for backward compatibility with Generate() calls in fan-out.
func flattenMessages(messages []llm.Message) string {
	var parts []string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			parts = append(parts, "Instructions:\n"+msg.Content)
		case "user":
			parts = append(parts, msg.Content)
		case "assistant":
			parts = append(parts, "Previous response:\n"+msg.Content)
		}
	}
	return strings.Join(parts, "\n\n")
}
