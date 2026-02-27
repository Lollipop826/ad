import argparse
import os
import sys
import time
import statistics


def _percentile(values, p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _summarize(name: str, times: list[float]) -> None:
    times_ms = [t * 1000.0 for t in times]
    print(f"\n== {name} ==")
    print(f"n={len(times_ms)}")
    print(f"mean={statistics.mean(times_ms):.1f}ms")
    print(f"p50={_percentile(times_ms, 50):.1f}ms")
    print(f"p90={_percentile(times_ms, 90):.1f}ms")
    print(f"p95={_percentile(times_ms, 95):.1f}ms")
    print(f"min={min(times_ms):.1f}ms")
    print(f"max={max(times_ms):.1f}ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tool", choices=["eval", "resist", "both"], default="both")
    parser.add_argument("--dimension", default="language")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.llm.model_pool import get_model_pool
    from src.tools.agent_tools.answer_evaluation_tool import AnswerEvaluationTool
    from src.tools.agent_tools.resistance_detection_tool import ResistanceDetectionTool

    print("[bench] init model pool...")
    _ = get_model_pool()

    eval_tool = AnswerEvaluationTool(use_local=True)
    resist_tool = ResistanceDetectionTool(use_local=True)

    question = "打游戏挺有意思的，最近在玩啥游戏呢，路洋？"
    answer_short = "刺激战场。"
    answer_long = "真可以借那个写字的东西，反正也得写，不费多少钱，我们不是兄弟吗？"

    def run_resist(a: str) -> float:
        t0 = time.perf_counter()
        _ = resist_tool._run(question=question, answer=a)
        return time.perf_counter() - t0

    def run_eval(a: str) -> float:
        t0 = time.perf_counter()
        _ = eval_tool._run(question=question, answer=a, dimension_id=args.dimension)
        return time.perf_counter() - t0

    for i in range(args.warmup):
        print(f"[bench] warmup {i+1}/{args.warmup}")
        if args.tool in ("resist", "both"):
            _ = run_resist(answer_short)
        if args.tool in ("eval", "both"):
            _ = run_eval(answer_short)

    resist_times_short: list[float] = []
    resist_times_long: list[float] = []
    eval_times_short: list[float] = []
    eval_times_long: list[float] = []

    for i in range(args.iters):
        print(f"[bench] iter {i+1}/{args.iters}")
        if args.tool in ("resist", "both"):
            resist_times_short.append(run_resist(answer_short))
            resist_times_long.append(run_resist(answer_long))
        if args.tool in ("eval", "both"):
            eval_times_short.append(run_eval(answer_short))
            eval_times_long.append(run_eval(answer_long))

    if args.tool in ("resist", "both"):
        _summarize("ResistanceTool short answer", resist_times_short)
        _summarize("ResistanceTool long answer", resist_times_long)
    if args.tool in ("eval", "both"):
        _summarize("AnswerEvalTool short answer", eval_times_short)
        _summarize("AnswerEvalTool long answer", eval_times_long)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
