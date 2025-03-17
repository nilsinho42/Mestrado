# Build stage
FROM golang:1.19-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy go.mod first
COPY go.mod ./

# Download and verify dependencies
RUN go mod download
RUN go mod verify

# Copy source code
COPY . .

# Ensure all dependencies are properly downloaded and verified
RUN go mod tidy
RUN go mod download
RUN go mod verify

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# Final stage
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache ca-certificates

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/main .

# Copy migrations
COPY --from=builder /app/migrations ./migrations

# Expose port 8080
EXPOSE 8080

# Run the application
CMD ["./main"] 