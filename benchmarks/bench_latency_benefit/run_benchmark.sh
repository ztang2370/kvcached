#!/bin/bash

# Parameter sweep benchmark for bench_kvcached_vllm.py
# Tests all combinations of: sending pattern x request rate x prompt length x generation length
# Runs 3 model instances in parallel with staggered start (model_delay).
#
# Sending patterns:
#   poisson  - Poisson process (exponential inter-arrival times, burstiness=1.0)
#   uniform  - Near-constant inter-arrival times (gamma distribution with burstiness=100.0)
#   ramp     - RPS increments by 1 each second: 1,2,...,peak,...,2,1,0
#
# NOTE: Ensure the vLLM server is configured with --max-model-len >= max(prompt) + max(gen)
#       (e.g., 32768 to support 16384 prompt + 2048 gen). Adjust gpu-memory-utilization
#       accordingly in bench-config.yaml.

trap 'echo ""; echo "Interrupted. Killing all benchmark clients..."; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM

export KVCACHED_IPC_NAME=VLLM
export PYTHONPATH="../../:../../benchmarks:$PYTHONPATH"

# ===================== Model Instances ===================
BACKEND="vllm"
ENDPOINT="/v1/completions"

MODELS=(
    "Qwen/Qwen3-8B:12346"
    "Qwen/Qwen3-8B:30000"
    "Qwen/Qwen3-8B:40000"
)
NUM_MODELS=${#MODELS[@]}

# ================= Sweep Parameters ======================
REQ_RATES=(5 10 20 30 40 50)
# PROMPT_LENS=(128 256 512 1024 2048 4096 8192 16384)
PROMPT_LENS=(128 256 512)
# GEN_LENS=(64 128 256 512 1024 2048)
GEN_LENS=(64 128 256 512 1024)
PATTERNS=("poisson" "uniform" "ramp")
# PATTERNS=("uniform" "ramp")


# Fixed number of requests for constant-rate patterns (poisson, uniform).
# Ramp pattern uses rate^2 (determined by its schedule shape).
NUM_REQUESTS=300

# ===================== Output ============================
RESULTS_DIR="results/sweep"
LOG_DIR="results/sweep/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ===================== Per-config runner =================
# Launches all 3 model instances with staggered start, waits for completion.
run_sweep_config() {
    local pattern=$1 rate=$2 prompt_len=$3 gen_len=$4
    local tag="${pattern}_rate${rate}_prompt${prompt_len}_gen${gen_len}"

    # Skip if ALL model results already exist
    local all_exist=true
    for i in "${!MODELS[@]}"; do
        local mi=$((i + 1))
        if [ ! -f "${RESULTS_DIR}/${tag}_model${mi}.json" ] || \
           [ ! -s "${RESULTS_DIR}/${tag}_model${mi}.json" ]; then
            all_exist=false
            break
        fi
    done
    if $all_exist; then
        return 2  # signal: skipped
    fi

    # Calculate base num_prompts and model_delay
    local base_num_prompts model_delay
    case "$pattern" in
        poisson|uniform)
            base_num_prompts=$NUM_REQUESTS
            # Delay = ~1/4 of expected duration (NUM_REQUESTS / rate seconds)
            model_delay=$(( NUM_REQUESTS / rate / 4 ))
            [ "$model_delay" -lt 5 ] && model_delay=5
            ;;
        ramp)
            base_num_prompts=$((rate * rate))
            # Original formula: ramp_up_duration/4 + ramp_up_duration*2
            # With start=0, peak=rate, increment=1: ramp_up_duration = rate
            model_delay=$(( rate / 4 + rate * 2 ))
            ;;
    esac
    [ "$base_num_prompts" -lt 10 ] && base_num_prompts=10

    local UNIFIED_START_TIME
    UNIFIED_START_TIME=$(date +%s.%N)
    echo "  Unified start: ${UNIFIED_START_TIME}, model_delay: ${model_delay}s"

    local PIDS=()
    local RESULT_FILES=()

    for i in "${!MODELS[@]}"; do
        local model_port="${MODELS[$i]}"
        local model="${model_port%%:*}"
        local port="${model_port##*:}"
        local mi=$((i + 1))
        local result_file="${RESULTS_DIR}/${tag}_model${mi}.json"
        local log_file="${LOG_DIR}/${tag}_model${mi}.log"

        RESULT_FILES+=("$result_file")

        # Stagger: delay before starting subsequent models
        if [ "$i" -gt 0 ] && [ "$model_delay" -gt 0 ]; then
            echo "  Waiting ${model_delay}s before starting model ${mi}..."
            sleep "$model_delay"
        fi

        # Extra prompts so earlier models keep sending while later models catch up.
        # For constant-rate patterns: extra = rate * delay * stagger_slots
        # For ramp: extra = delay * stagger_slots (avg RPS ~ 1 during ramp-up)
        local stagger_extra
        local stagger_slots=$(( NUM_MODELS - 1 - i ))
        case "$pattern" in
            poisson|uniform) stagger_extra=$(( rate * model_delay * stagger_slots )) ;;
            ramp)            stagger_extra=$(( model_delay * stagger_slots )) ;;
        esac
        local num_prompts=$((base_num_prompts + stagger_extra))
        [ "$num_prompts" -lt 10 ] && num_prompts=10

        local common_args=(
            --backend "$BACKEND"
            --model "$model"
            --dataset-name random
            --random-input-len "$prompt_len"
            --random-output-len "$gen_len"
            --num-prompts "$num_prompts"
            --host "localhost"
            --port "$port"
            --endpoint "$ENDPOINT"
            --save-result
            --result-filename "$result_file"
            --metadata "unified_start_time=$UNIFIED_START_TIME" \
                       "pattern=$pattern" "req_rate=$rate" \
                       "prompt_len=$prompt_len" "gen_len=$gen_len" \
                       "model_index=$mi" "num_models=$NUM_MODELS" \
                       "model_delay=$model_delay"
        )

        echo "  Starting model ${mi} on port ${port} (n=${num_prompts})..."

        case "$pattern" in
            poisson)
                python bench_kvcached_vllm.py "${common_args[@]}" \
                    --request-rate "$rate" \
                    --burstiness 1.0 \
                    >"$log_file" 2>&1 &
                ;;
            uniform)
                python bench_kvcached_vllm.py "${common_args[@]}" \
                    --request-rate "$rate" \
                    --burstiness 100.0 \
                    >"$log_file" 2>&1 &
                ;;
            ramp)
                python bench_kvcached_vllm.py "${common_args[@]}" \
                    --ramp-up-strategy ramp-up-down \
                    --ramp-start-rps 0 \
                    --ramp-peak-rps "$rate" \
                    --ramp-end-rps 0 \
                    --ramp-increment 1 \
                    >"$log_file" 2>&1 &
                ;;
        esac

        PIDS+=($!)
        echo "  Model ${mi} started with PID ${PIDS[$i]}"
    done

    # Wait for all models to finish
    echo "  Waiting for all ${NUM_MODELS} models to complete..."
    local all_ok=true
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        local ec=$?
        local mi=$((i + 1))
        if [ $ec -ne 0 ]; then
            all_ok=false
            echo "  Model ${mi} FAILED (exit=${ec}, see ${LOG_DIR}/${tag}_model${mi}.log)"
        else
            echo "  Model ${mi} OK"
        fi
    done

    echo "  Results:"
    for rf in "${RESULT_FILES[@]}"; do
        echo "    - ${rf}"
    done

    $all_ok && return 0 || return 1
}

# ===================== Main sweep ========================
total=$(( ${#PATTERNS[@]} * ${#REQ_RATES[@]} * ${#PROMPT_LENS[@]} * ${#GEN_LENS[@]} ))

echo "========================================"
echo "  Benchmark Parameter Sweep (${NUM_MODELS} instances)"
echo "========================================"
echo "Models:"
for m in "${MODELS[@]}"; do
    echo "  - ${m}"
done
echo "Rates:    ${REQ_RATES[*]}"
echo "Prompts:  ${PROMPT_LENS[*]}"
echo "Gen:      ${GEN_LENS[*]}"
echo "Patterns: ${PATTERNS[*]}"
echo "Total:    ${total} configurations (x${NUM_MODELS} models each)"
echo "Output:   ${RESULTS_DIR}/"
echo "========================================"
echo ""

ok=0; fail=0; skip=0; count=0

for pattern in "${PATTERNS[@]}"; do
  for rate in "${REQ_RATES[@]}"; do
    for prompt_len in "${PROMPT_LENS[@]}"; do
      for gen_len in "${GEN_LENS[@]}"; do
        count=$((count + 1))
        tag="${pattern}_rate${rate}_prompt${prompt_len}_gen${gen_len}"

        echo "[${count}/${total}] ${tag}"

        run_sweep_config "$pattern" "$rate" "$prompt_len" "$gen_len"
        rc=$?

        if [ $rc -eq 2 ]; then
            skip=$((skip + 1))
            echo "  SKIP (all model results exist)"
        elif [ $rc -eq 0 ]; then
            ok=$((ok + 1))
            echo "  DONE"
        else
            fail=$((fail + 1))
            echo "  FAIL (one or more models failed)"
        fi

        echo "  Progress: $((ok + fail + skip))/${total}  (ok=${ok} fail=${fail} skip=${skip})"
        echo ""
      done
    done
  done
done

echo "========================================"
echo "  Sweep Complete"
echo "========================================"
echo "Succeeded: $ok"
echo "Failed:    $fail"
echo "Skipped:   $skip"
echo "Total:     $total"
echo "Results:   $RESULTS_DIR/"
echo "========================================"
