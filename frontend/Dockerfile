# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Create env file from build args
ARG VITE_API_URL
ARG VITE_WS_URL
ARG VITE_APP_TITLE
ARG VITE_MLFLOW_URL
ARG VITE_GRAFANA_URL

# Set environment variables
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_APP_TITLE=$VITE_APP_TITLE
ENV VITE_MLFLOW_URL=$VITE_MLFLOW_URL
ENV VITE_GRAFANA_URL=$VITE_GRAFANA_URL

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Add support for host.docker.internal
RUN apk add --no-cache bash

# Create entrypoint script
RUN printf '#!/bin/bash\n\
echo "Updating hosts file..."\n\
DOCKER_INTERNAL_IP=$(getent hosts host.docker.internal | cut -d" " -f1)\n\
if [ -z "$DOCKER_INTERNAL_IP" ]; then\n\
    echo "172.17.0.1 host.docker.internal" >> /etc/hosts\n\
fi\n\
echo "Starting nginx..."\n\
exec nginx -g "daemon off;"\n' > /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx using the entrypoint script
ENTRYPOINT ["/entrypoint.sh"] 