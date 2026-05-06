FROM node:20-alpine

WORKDIR /app

# Install dependencies first (better caching)
COPY package.json package-lock.json* yarn.lock* ./
RUN npm install

# Copy source files
COPY . .

# Expose dev server port
EXPOSE 8080

# Default to dev server
CMD ["npm", "run", "dev"]
