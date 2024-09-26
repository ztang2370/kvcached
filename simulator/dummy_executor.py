from typing import List, Set, Tuple

from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput

logger = init_logger(__name__)


class DummyExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        return

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return (8000, 8000)  # TODO(yifan): update this

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        return

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        return None

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return True

    def remove_lora(self, lora_id: int) -> bool:
        return True

    def list_loras(self) -> Set[int]:
        return []

    def check_health(self) -> None:
        return
