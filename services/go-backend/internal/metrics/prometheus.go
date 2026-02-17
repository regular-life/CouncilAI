package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RequestCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "padhai_dost_request_count_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "path", "status"},
	)

	LatencyHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "padhai_dost_request_latency_seconds",
			Help:    "Request latency in seconds",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 12),
		},
		[]string{"method", "path"},
	)

	ChairmanSynthesisCount = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "padhai_dost_chairman_synthesis_count_total",
			Help: "Total number of chairman synthesis invocations",
		},
	)

	LLMFailureCount = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "padhai_dost_llm_failure_count_total",
			Help: "Total number of LLM call failures",
		},
	)

	CacheHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "padhai_dost_cache_operations_total",
			Help: "Total cache operations by result",
		},
		[]string{"result"},
	)

	CouncilResponseTime = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "padhai_dost_council_response_seconds",
			Help:    "Council orchestration total response time",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
		},
	)
)
