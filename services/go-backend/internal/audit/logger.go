package audit

import (
	"encoding/json"
	"log"
	"time"
)

type Entry struct {
	UserID    string        `json:"user_id"`
	DocID     string        `json:"doc_id"`
	QueryHash string       `json:"query_hash"`
	Action    string        `json:"action"`
	Timestamp time.Time     `json:"timestamp"`
	Latency   time.Duration `json:"latency_ms"`
	Status    string        `json:"status"`
	Details   string        `json:"details,omitempty"`
}

type Logger struct{}

func NewLogger() *Logger {
	return &Logger{}
}

func (l *Logger) Log(entry Entry) {
	entry.Timestamp = time.Now()
	data, err := json.Marshal(entry)
	if err != nil {
		log.Printf("[Audit] Failed to marshal entry: %v", err)
		return
	}
	log.Printf("[Audit] %s", string(data))
}

func (l *Logger) LogQuery(userID, docID, queryHash string, latency time.Duration, status string) {
	l.Log(Entry{
		UserID:    userID,
		DocID:     docID,
		QueryHash: queryHash,
		Action:    "query",
		Latency:   latency,
		Status:    status,
	})
}

func (l *Logger) LogIngest(userID, docID string, latency time.Duration, status string) {
	l.Log(Entry{
		UserID:  userID,
		DocID:   docID,
		Action:  "ingest",
		Latency: latency,
		Status:  status,
	})
}
