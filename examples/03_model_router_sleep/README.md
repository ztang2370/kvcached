# KVCached with vLLM Semantic Router Example

This example demonstrates how to integrate KVCached with the [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) for intelligent model routing based on query content. The semantic router automatically routes requests to the most appropriate model endpoint based on semantic analysis of the input using BERT-based classification.

The vLLM Semantic Router is an intelligent Mixture-of-Models (MoM) router that directs OpenAI API requests to the most suitable models from a defined pool based on semantic understanding of the request's intent, complexity, and task type.

## Overview

The setup consists of:
- **vLLM Semantic Router**: Intelligent Mixture-of-Models router with BERT-based classification for semantic routing
- **Two vLLM servers**: Running different models with KVCached integration for efficient GPU memory sharing
- **Envoy proxy**: Handles request routing and load balancing with external processing
- **Monitoring stack**: Prometheus metrics collection and Grafana dashboards for observability
- **Security features**: PII detection and prompt guard for enterprise-grade security

The semantic router provides additional enterprise features like:
- **Auto-reasoning**: Selects reasoning capabilities based on query complexity
- **Tool selection**: Automatically chooses relevant tools to reduce prompt tokens
- **PII detection**: Protects user privacy by detecting personally identifiable information
- **Prompt guard**: Prevents jailbreak attacks and malicious prompts
- **Similarity caching**: Caches semantic representations to improve latency

## Prerequisites

- Docker and Docker Compose v2 (`docker compose` command)
- GPU with sufficient memory for two Qwen models
- vLLM virtual environment with KVCached integration at `../../engine_integration/vllm-v0.9.2/.venv` (relative to this examples directory)

## Quickstart

### 1. Clone and Setup Semantic Router

```bash
./setup_semantic_router.sh --model-a "Qwen/Qwen2.5-1.5B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"
```

This script will:
- Clone the semantic router repository
- Apply necessary patches for Docker and configuration with your specified models
- Set up the required configuration files

**Model Configuration Options:**
- `--model-a MODEL`: Model for endpoint1 (port 12346) - handles business, law, psychology, biology, chemistry, history, health, and general queries
- `--model-b MODEL`: Model for endpoint2 (port 12347) - specializes in economics, math, physics, computer science, philosophy, and engineering

**Examples:**

```bash
# Use different Qwen models
./setup_semantic_router.sh --model-a "Qwen/Qwen2.5-3B-Instruct" --model-b "Qwen/Qwen2.5-7B-Instruct"

# Use Llama models
./setup_semantic_router.sh --model-a "meta-llama/Llama-3.2-3B-Instruct" --model-b "meta-llama/Llama-3.2-1B-Instruct"

# Use environment variables
MODEL_A="microsoft/DialoGPT-medium" MODEL_B="facebook/opt-1.3b" ./setup_semantic_router.sh
```

### 2. Start the Semantic Router Stack

```bash
cd semantic-router
docker compose up --build
```

This starts:
- Semantic router service
- Envoy proxy (port 8801)
- Prometheus monitoring
- Grafana dashboard (port 4000, admin/admin)

### 3. Start vLLM Servers with KVCached

**Option A: Using the convenience script (recommended):**

```bash
./start_vllm_servers.sh --model-a "Qwen/Qwen2.5-1.5B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"
```

**Model Configuration Options:**
- `--model-a MODEL`: Model for endpoint1 (port 12346)
- `--model-b MODEL`: Model for endpoint2 (port 12347)

**Examples:**

```bash
# Use custom models
./start_vllm_servers.sh --model-a "meta-llama/Llama-3.2-3B-Instruct" --model-b "microsoft/DialoGPT-small"

# Use environment variables
MODEL_A="facebook/opt-1.3b" MODEL_B="distilbert-base-uncased" ./start_vllm_servers.sh
```

**Option B: Manual startup in separate terminals:**

**Terminal 1:**

```bash
source ../../engine_integration/vllm-v0.9.2/.venv/bin/activate
export ENABLE_KVCACHED=true
vllm serve YOUR_MODEL_A \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --port=12346 \
  --tensor-parallel-size=1
```

**Terminal 2:**

```bash
source ../../engine_integration/vllm-v0.9.2/.venv/bin/activate
export ENABLE_KVCACHED=true
vllm serve YOUR_MODEL_B \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --port=12347 \
  --tensor-parallel-size=1
```

### 4. Test the Semantic Router

**Option A: Using the comprehensive test script (recommended):**

```bash
./test_routing.sh --model-a-name "Qwen2.5-1.5B-Instruct" --model-b-name "Qwen1.5-1.8B"
```

This will test various query types across different categories to verify routing works correctly.

**Test Script Options:**
- `--model-a-name NAME`: Expected model name for endpoint1
- `--model-b-name NAME`: Expected model name for endpoint2

**Example with custom models:**

```bash
./test_routing.sh --model-a-name "Llama-3.2-3B-Instruct" --model-b-name "Qwen1.5-1.8B"
```

**Option B: Manual testing with curl:**

Send a request to the semantic router:

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of x^4?"}
    ]
  }'
```

The semantic router will analyze the mathematical content and route to the Qwen1.5-1.8B model (optimized for math/physics).

## Usage Examples

### Complete Setup with Scripts (Recommended)

```bash
# 1. Setup semantic router with custom models
./setup_semantic_router.sh --model-a "meta-llama/Llama-3.2-3B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"

# 2. Start semantic router stack
cd semantic-router
docker compose up --build

# 3. Start vLLM servers (in new terminal, back in examples directory)
cd ../examples/03_model_router_sleep
# The convenience script automatically activates the vLLM KVCached virtual environment
./start_vllm_servers.sh --model-a "meta-llama/Llama-3.2-3B-Instruct" --model-b "Qwen/Qwen1.5-1.8B"

# 4. Test routing
./test_routing.sh --model-a-name "meta-llama/Llama-3.2-3B-Instruct" --model-b-name "Qwen/Qwen1.5-1.8B"
```

### Using Environment Variables

```bash
# Set models via environment variables
export MODEL_A="microsoft/DialoGPT-medium"
export MODEL_B="facebook/opt-1.3b"

# Setup and start with environment variables
./setup_semantic_router.sh
cd semantic-router
docker compose up --build
cd ../examples/03_model_router_sleep
# The convenience script automatically activates the vLLM KVCached virtual environment
./start_vllm_servers.sh  # Uses MODEL_A and MODEL_B environment variables
./test_routing.sh  # Uses MODEL_A_NAME and MODEL_B_NAME environment variables
```

### Manual Setup

```bash
# 1. Setup semantic router with specific models
./setup_semantic_router.sh --model-a "YOUR_MODEL_A" --model-b "YOUR_MODEL_B"

# 2. Start semantic router stack
cd semantic-router
docker compose up --build

# 3. Start vLLM servers manually in separate terminals
source ../../engine_integration/vllm-v0.9.2/.venv/bin/activate
export ENABLE_KVCACHED=true
vllm serve YOUR_MODEL_A --disable-log-requests --no-enable-prefix-caching --port=12346 --tensor-parallel-size=1 &
vllm serve YOUR_MODEL_B --disable-log-requests --no-enable-prefix-caching --port=12347 --tensor-parallel-size=1 &

# 4. Test manually
curl -X POST http://localhost:8801/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the derivative of x^4?"}]}'
```

## Model Routing Logic

The semantic router uses category classification to route requests:

- **Qwen2.5-1.5B-Instruct**: Optimized for business, law, psychology, biology, chemistry, history, health, and general queries
- **Qwen1.5-1.8B**: Specialized for economics, math, physics, computer science, philosophy, and engineering

## Monitoring

- **Grafana Dashboard**: http://localhost:4000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Configuration

The semantic router configuration is in `config/config.yaml` and includes:
- Model endpoints and ports
- Category-based routing rules with model scores
- BERT-based semantic classification
- PII detection and prompt guard settings
- Auto-reasoning configuration per model

For detailed configuration options and advanced features, see the [vLLM Semantic Router Documentation](https://vllm-semantic-router.readthedocs.io/).

## Troubleshooting

1. **Port conflicts**: Ensure ports 12346, 12347, 8801, 4000 are available
2. **GPU memory**: Monitor GPU usage with `nvidia-smi`
3. **KVCached**: Verify `ENABLE_KVCACHED=true` is set
4. **Docker networks**: Check that vLLM servers can communicate with the semantic router

## Scripts

- `setup_semantic_router.sh`: Clone and patch semantic router with all necessary configuration changes
- `start_vllm_servers.sh`: Convenience script to start both vLLM servers with KVCached enabled in the background
- `test_routing.sh`: Comprehensive testing script that sends various queries to verify semantic routing across different categories
- `docker-compose.yml`: Docker Compose file for the semantic router stack (located in the semantic-router directory after setup, uses `docker compose` command)
