package middleware

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/regular-life/padhai-dost/go-backend/internal/metrics"
)

type statusWriter struct {
	http.ResponseWriter
	statusCode int
	written    bool
}

func (w *statusWriter) WriteHeader(code int) {
	if w.written {
		return
	}
	w.written = true
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(sw, r)
		duration := time.Since(start)

		log.Printf("[HTTP] %s %s → %d (%s)", r.Method, r.URL.Path, sw.statusCode, duration)
		metrics.RequestCount.WithLabelValues(r.Method, r.URL.Path, fmt.Sprintf("%d", sw.statusCode)).Inc()
		metrics.LatencyHistogram.WithLabelValues(r.Method, r.URL.Path).Observe(duration.Seconds())
	})
}
