#!/bin/bash

set -e

# Default models
MODEL_A="${MODEL_A:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_B="${MODEL_B:-Qwen/Qwen1.5-1.8B}"

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup semantic router with configurable models for KVCached integration"
    echo ""
    echo "Options:"
    echo "  --model-a MODEL     Model for endpoint1 (port 12346) [default: $MODEL_A]"
    echo "  --model-b MODEL     Model for endpoint2 (port 12347) [default: $MODEL_B]"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_A             Model for endpoint1 (can be overridden by --model-a)"
    echo "  MODEL_B             Model for endpoint2 (can be overridden by --model-b)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --model-a 'meta-llama/Llama-3.2-1B' --model-b 'microsoft/DialoGPT-small'"
    echo "  MODEL_A='mistralai/Mistral-7B-v0.1' MODEL_B='facebook/opt-1.3b' $0"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-a)
            MODEL_A="$2"
            shift 2
            ;;
        --model-b)
            MODEL_B="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "Setting up Semantic Router with KVCached integration..."
echo "Model A (endpoint1): $MODEL_A"
echo "Model B (endpoint2): $MODEL_B"

# Clone semantic router
if [ ! -d "semantic-router" ]; then
    echo "Cloning semantic router..."
    git clone https://github.com/vllm-project/semantic-router.git
else
    echo "Semantic router already exists, skipping clone..."
fi

cd semantic-router

# Create config directory if it doesn't exist
mkdir -p config

# Apply patches using the diff provided
echo "Applying patches..."

# Patch 1: Dockerfile.extproc - Add -mod=vendor to go build
cat > dockerfile_extproc.patch << 'EOF'
diff --git a/Dockerfile.extproc b/Dockerfile.extproc
index 72ead6e..169e2a0 100644
--- a/Dockerfile.extproc
+++ b/Dockerfile.extproc
@@ -40,7 +40,7 @@ ENV CGO_ENABLED=1
 ENV LD_LIBRARY_PATH=/app/candle-binding/target/release

 # Build the router binary
-RUN mkdir -p bin && cd src/semantic-router && go build -o ../../bin/router cmd/main.go
+RUN mkdir -p bin && cd src/semantic-router && go build -mod=vendor -o ../../bin/router cmd/main.go

 # Final stage: copy the binary and the shared library
 FROM quay.io/centos/centos:stream9
EOF
patch -p1 < dockerfile_extproc.patch

# Update config/config.yaml - Replace with our modified version
echo "Updating config/config.yaml with custom models..."

# Create backup of original config
cp config/config.yaml config/config.yaml.backup

# Replace the config with our modified version
cat > config/config.yaml << EOF
bert_model:
  model_id: /app/models/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

semantic_cache:
  enabled: true
  backend_type: "memory"  # Options: "memory" or "milvus"
  similarity_threshold: 0.8
  max_entries: 1000  # Only applies to memory backend
  ttl_seconds: 3600
  eviction_policy: "fifo"

tools:
  enabled: true
  top_k: 3
  similarity_threshold: 0.2
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true

prompt_guard:
  enabled: true
  use_modernbert: true
  model_id: "/app/models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7
  use_cpu: true
  jailbreak_mapping_path: "models/jailbreak_classifier_modernbert-base_model/jailbreak_type_mapping.json"

# vLLM Endpoints Configuration
# IMPORTANT: 'address' field must be a valid IP address (IPv4 or IPv6)
# Supported formats: 127.0.0.1, 192.168.1.1, ::1, 2001:db8::1
# NOT supported: domain names (example.com), protocol prefixes (http://), paths (/api), ports in address (use 'port' field)

vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"  # IPv4 address - REQUIRED format
    port: 12346
    models:
      - "$MODEL_A"
    weight: 1
  - name: "endpoint2"
    address: "127.0.0.1"
    port: 12347
    models:
      - "$MODEL_B"
    weight: 1

model_config:
  "$MODEL_A":
    preferred_endpoints: ["endpoint1"]
    pii_policy:
      allow_by_default: true
  "$MODEL_B":
    preferred_endpoints: ["endpoint2"]
    pii_policy:
      allow_by_default: true

# Classifier configuration
classifier:
  category_model:
    model_id: "/app/models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
  pii_model:
    model_id: "/app/models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"

# Categories with new use_reasoning field structure
categories:
  - name: business
    model_scores:
      - model: $MODEL_A
        score: 0.7
        use_reasoning: false  # Business performs better without reasoning
  - name: law
    model_scores:
      - model: $MODEL_A
        score: 0.4
        use_reasoning: false
  - name: psychology
    model_scores:
      - model: $MODEL_A
        score: 0.6
        use_reasoning: false
  - name: biology
    model_scores:
      - model: $MODEL_A
        score: 0.9
        use_reasoning: false
  - name: chemistry
    model_scores:
      - model: $MODEL_A
        score: 0.6
        use_reasoning: true  # Enable reasoning for complex chemistry
  - name: history
    model_scores:
      - model: $MODEL_A
        score: 0.7
        use_reasoning: false
  - name: other
    model_scores:
      - model: $MODEL_A
        score: 0.7
        use_reasoning: false
  - name: health
    model_scores:
      - model: $MODEL_A
        score: 0.5
        use_reasoning: false
  - name: economics
    model_scores:
      - model: $MODEL_B
        score: 1.0
        use_reasoning: false
  - name: math
    model_scores:
      - model: $MODEL_B
        score: 1.0
        use_reasoning: true  # Enable reasoning for complex math
  - name: physics
    model_scores:
      - model: $MODEL_B
        score: 0.7
        use_reasoning: true  # Enable reasoning for physics
  - name: computer science
    model_scores:
      - model: $MODEL_B
        score: 0.6
        use_reasoning: false
  - name: philosophy
    model_scores:
      - model: $MODEL_B
        score: 0.5
        use_reasoning: false
  - name: engineering
    model_scores:
      - model: $MODEL_B
        score: 0.7
        use_reasoning: false

default_model: $MODEL_A

# Reasoning family configurations
reasoning_families:
  deepseek:
    type: "chat_template_kwargs"
    parameter: "thinking"

  qwen3:
    type: "chat_template_kwargs"
    parameter: "enable_thinking"

  gpt-oss:
    type: "reasoning_effort"
    parameter: "reasoning_effort"
  gpt:
    type: "reasoning_effort"
    parameter: "reasoning_effort"

# Global default reasoning effort level
default_reasoning_effort: high

# API Configuration
api:
  batch_classification:
    max_batch_size: 100
    concurrency_threshold: 5
    max_concurrency: 8
    metrics:
      enabled: true
      detailed_goroutine_tracking: true
      high_resolution_timing: false
      sample_rate: 1.0
      duration_buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]
      size_buckets: [1, 2, 5, 10, 20, 50, 100, 200]
EOF

# Patch 3: config/envoy-docker.yaml - Update routing and add dynamic forward proxy
cat > envoy_docker_yaml.patch << 'EOF'
diff --git a/config/envoy-docker.yaml b/config/envoy-docker.yaml
index 2700b49..3837b09 100644
--- a/config/envoy-docker.yaml
+++ b/config/envoy-docker.yaml
@@ -38,11 +38,11 @@ static_resources:
             - name: local_service
               domains: ["*"]
               routes:
-              # Single route using original destination cluster
+              # Single route using dynamic forward proxy
               - match:
                   prefix: "/"
                 route:
-                  cluster: vllm_dynamic_cluster
+                  cluster: dynamic_forward_proxy_cluster
                   timeout: 300s
           http_filters:
           - name: envoy.filters.http.ext_proc
@@ -61,6 +61,22 @@ static_resources:
                 response_trailer_mode: "SKIP"
               failure_mode_allow: true
               message_timeout: 300s
+          - name: envoy.filters.http.lua
+            typed_config:
+              "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
+              inline_code: |
+                function envoy_on_request(request_handle)
+                  local endpoint = request_handle:headers():get("x-semantic-destination-endpoint")
+                  if endpoint ~= nil then
+                    request_handle:headers():replace("host", endpoint)
+                  end
+                end
+          - name: envoy.filters.http.dynamic_forward_proxy
+            typed_config:
+              "@type": type.googleapis.com/envoy.extensions.filters.http.dynamic_forward_proxy.v3.FilterConfig
+              dns_cache_config:
+                name: dynamic_forward_proxy_cache_config
+                dns_lookup_family: ALL
           - name: envoy.filters.http.router
             typed_config:
               "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
@@ -98,20 +114,18 @@ static_resources:
                 address: semantic-router  # Use Docker service name
                 port_value: 50051

-  # Dynamic vLLM cluster using original destination
-  - name: vllm_dynamic_cluster
+  # Dynamic forward proxy cluster
+  - name: dynamic_forward_proxy_cluster
     connect_timeout: 300s
     per_connection_buffer_limit_bytes: 52428800
-    type: ORIGINAL_DST
     lb_policy: CLUSTER_PROVIDED
-    original_dst_lb_config:
-      use_http_header: true
-      http_header_name: "x-semantic-destination-endpoint"
-    typed_extension_protocol_options:
-      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
-        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
-        explicit_http_config:
-          http_protocol_options: {}
+    cluster_type:
+      name: envoy.clusters.dynamic_forward_proxy
+      typed_config:
+        "@type": type.googleapis.com/envoy.extensions.clusters.dynamic_forward_proxy.v3.ClusterConfig
+        dns_cache_config:
+          name: dynamic_forward_proxy_cache_config
+          dns_lookup_family: ALL

 admin:
   address:
EOF
patch -p1 < envoy_docker_yaml.patch

# Patch 4: docker-compose.yml - Change Grafana port
cat > docker_compose_yml.patch << 'EOF'
diff --git a/docker-compose.yml b/docker-compose.yml
index 2f9931e..369d8a7 100644
--- a/docker-compose.yml
+++ b/docker-compose.yml
@@ -85,7 +85,7 @@ services:
       - GF_SECURITY_ADMIN_USER=admin
       - GF_SECURITY_ADMIN_PASSWORD=admin
     ports:
-      - "3000:3000"
+      - "4000:3000"
     volumes:
       - ./config/grafana/datasource.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
       - ./config/grafana/dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
EOF
patch -p1 < docker_compose_yml.patch

# Required Patch 5: src/semantic-router/pkg/extproc/request_handler.go - Update endpoint mapping for Docker
echo "Applying required Docker gateway IP mapping patch..."
cat > request_handler_go.patch << 'EOF'
--- a/src/semantic-router/pkg/extproc/request_handler.go
+++ b/src/semantic-router/pkg/extproc/request_handler.go
@@ -393,6 +393,12 @@
 				endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(matchedModel)
 				if endpointFound {
 					selectedEndpoint = endpointAddress
+					// Map to Docker gateway IP but keep the port
+					if parts := strings.Split(endpointAddress, ":"); len(parts) == 2 {
+						selectedEndpoint = "172.17.0.1:" + parts[1]
+					} else {
+						selectedEndpoint = "172.17.0.1"
+					}
 					observability.Infof("Selected endpoint address: %s for model: %s", selectedEndpoint, matchedModel)
 				} else {
 					observability.Warnf("No endpoint found for model %s, using fallback", matchedModel)
@@ -438,7 +444,7 @@
 				if actualModel != "" {
 					setHeaders = append(setHeaders, &core.HeaderValueOption{
 						Header: &core.HeaderValue{
 							Key:      "x-selected-model",
-							Value:    actualModel,
+							// Value:    actualModel,
 							RawValue: []byte(actualModel),
 						},
 					})
@@ -508,6 +514,12 @@
 			endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(originalModel)
 			if endpointFound {
 				selectedEndpoint = endpointAddress
+				// Map to Docker gateway IP but keep the port
+				if parts := strings.Split(endpointAddress, ":"); len(parts) == 2 {
+					selectedEndpoint = "172.17.0.1:" + parts[1]
+				} else {
+					selectedEndpoint = "172.17.0.1"
+				}
 				observability.Infof("Selected endpoint address: %s for model: %s", selectedEndpoint, originalModel)
 			} else {
 				observability.Warnf("No endpoint found for model %s, using fallback", originalModel)
EOF

# Apply the patch - this is required for Docker networking
if patch -p1 < request_handler_go.patch; then
    echo "✓ Docker gateway IP mapping patch applied successfully"
else
    echo "✗ Docker gateway IP mapping patch failed - this is required for your setup"
    echo "  Please check the patch manually or update the line numbers"
    exit 1
fi

# Clean up patch files
rm -f dockerfile_extproc.patch config_yaml.patch envoy_docker_yaml.patch docker_compose_yml.patch request_handler_go.patch

echo "Semantic router setup complete!"
echo ""
echo "Next steps:"
echo "1. Start the semantic router services:"
echo "   cd semantic-router"
echo "   docker compose up --build"
echo ""
echo "2. In separate terminals, start vLLM servers (activate vLLM KVCached venv first):"
echo "   source ../../engine_integration/vllm-v0.9.2/.venv/bin/activate"
echo "   export ENABLE_KVCACHED=true"
echo "   vllm serve $MODEL_A --disable-log-requests --no-enable-prefix-caching --port=12346 --tensor-parallel-size=1"
echo "   vllm serve $MODEL_B --disable-log-requests --no-enable-prefix-caching --port=12347 --tensor-parallel-size=1"
echo "   OR use the convenience script: ./start_vllm_servers.sh --model-a '$MODEL_A' --model-b '$MODEL_B'"
echo ""
echo "3. Test with: curl -X POST http://localhost:8801/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the derivative of x^4?\"}]}'"
