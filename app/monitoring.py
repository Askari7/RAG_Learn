from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenCostEstimator:
    """
    Tracks token usage and estimates LLM API cost.
    """

    def __init__(
        self,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
    ):
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def calculate_cost(self) -> float:
        input_cost = (
            self.total_input_tokens / 1_000_000
        ) * self.input_cost_per_1m

        output_cost = (
            self.total_output_tokens / 1_000_000
        ) * self.output_cost_per_1m

        return input_cost + output_cost

    def get_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
        )

    def report(self) -> dict:
        usage = self.get_usage()

        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost_usd": round(
                self.calculate_cost(), 6
            ),
        }

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
