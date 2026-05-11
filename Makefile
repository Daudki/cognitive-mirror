.PHONY: help install train run test clean docker-build docker-up docker-down

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  [36m%-15s[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install -r requirements.txt

train:  ## Train the ML models
	python ml/train.py

run:  ## Run development server
	python app.py

test:  ## Run test suite
	python -m pytest tests/ -v

lint:  ## Check code style
	python -m ruff check cognitive_mirror/

format:  ## Format code with ruff
	python -m ruff format cognitive_mirror/

clean:  ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker image
	docker build -t cognitive-mirror:latest .

docker-up:  ## Start with Docker Compose
	docker-compose up -d

docker-down:  ## Stop Docker Compose
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f app

health:  ## Check API health
	@curl -s http://localhost:5000/api/v1/health | python -m json.tool

predict:  ## Test prediction endpoint
	@curl -s -X POST http://localhost:5000/api/v1/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "I am feeling very curious and excited about this project!"}' \
		| python -m json.tool

benchmark:  ## Run load test (requires Apache Bench)
	@echo '{"text": "test benchmark"}' > /tmp/bench.json
	ab -n 1000 -c 10 -p /tmp/bench.json -T "application/json" http://localhost:5000/api/v1/predict
	@rm /tmp/bench.json
