"""Count prefill CUDA graph captures and replays without synchronizing the GPU."""


def install():
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner as Runner
    capture, replay = Runner.capture, Runner.replay
    metrics = {'captures': 0, 'replays': 0, 'replayed_tokens': 0, 'buckets': {}}

    def captured(runner, num_tokens, forward):
        result = capture(runner, num_tokens, forward)
        metrics['captures'] += 1
        metrics['buckets'].setdefault(str(num_tokens), 0)
        return result

    def replayed(runner, num_tokens):
        result = replay(runner, num_tokens)
        metrics['replays'] += 1
        metrics['replayed_tokens'] += num_tokens
        metrics['buckets'][str(num_tokens)] = metrics['buckets'].get(str(num_tokens), 0) + 1
        return result

    Runner.capture, Runner.replay = captured, replayed

    def restore():
        Runner.capture, Runner.replay = capture, replay
    return metrics, restore
