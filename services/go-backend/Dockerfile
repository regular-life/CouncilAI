FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum* ./
RUN go mod download

# Copy source
COPY . .

# Build
RUN apk add --no-cache build-base
ENV CGO_CXXFLAGS_ALLOW="-mavx2|-mfma"
RUN CGO_ENABLED=1 GOOS=linux go build -o /server ./cmd/server

# Runtime stage
FROM alpine:3.19

RUN apk --no-cache add ca-certificates libstdc++

WORKDIR /app
COPY --from=builder /server .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1

CMD ["./server"]
