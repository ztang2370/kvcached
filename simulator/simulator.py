from typing import List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoConfig

from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter

from simulator.llm_engine import LLMEngine


class Simulator:

    def __init__(self, **kwargs):
        model_name = "meta-llama/Meta-Llama-3-8B"
        self.load_model("meta-llama/Meta-Llama-3-8B")
        self.dataset = self.load_dataset("todo")

        engine_args = EngineArgs(
            model=model_name,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def load_model(self, model: str):
        self.model_config: AutoConfig = AutoConfig.from_pretrained(model)
        # self.kv_size = self.model_config.hidden_size * self.model_config.num_kv_heads * self.model_config.num_layers

    def load_dataset(self, dataset: str) -> List[Tuple[int, int]]:
        # return [(10, 10), (100, 100), (1000, 1000), (10000, 10000)]
        return [(1024, 128)] * 1000

    def run(self):
        use_tqdm = True
        prompt_token_ids = []
        sampling_params = []
        for input_len, output_len in self.dataset:
            prompt_token_ids.append(dummy_tokenizer(input_len))
            sampling_params.append(SamplingParams(max_tokens=output_len))
        num_requests = len(prompt_token_ids)

        # Add requests to the engine.
        for i in range(num_requests):
            self._add_request(sampling_params[i], prompt_token_ids[i])
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    None,  # prompt
                                    sampling_params,
                                    prompt_token_ids)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="Processed prompts",
                        dynamic_ncols=True)
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs


def dummy_tokenizer(input_len: int):
    # [1] is the <s> token. and [7251] is the token for "hi"
    prompt_id = [1] + [7251] * (input_len - 1)
    return prompt_id
