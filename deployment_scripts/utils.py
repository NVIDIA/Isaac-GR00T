import torch
import argparse
import time
import numpy as np


def benchmark_policy_cuda_event(model, arg_inputs, kwarg_inputs, args: argparse.Namespace = None):
    """
    Benchmark the policy inference performance using CUDA events.
    
    Args:
        policy: The Groot policy to benchmark
        step_data: The input data for inference
        num_iterations: Number of iterations to run for benchmarking
        warmup_iterations: Number of warmup iterations before benchmarking
    
    Returns:
        Dict containing benchmark results
    """
    
    with torch.inference_mode(), torch.no_grad():
        print(f"Running warmup for {args.warmup_iterations} iterations...")
        # Warmup iterations
        for i in range(args.warmup_iterations):
            _ = model(*arg_inputs, **kwarg_inputs)
        
        # Make sure warmup is done
        torch.cuda.synchronize()
        
        print(f"Running benchmark for {args.num_iterations} iterations...")
        # CUDA events for timing
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        # Benchmark iterations
        starter.record()
        for i in range(args.num_iterations):
            model(*arg_inputs, **kwarg_inputs)

        ender.record()

        torch.cuda.synchronize()
        elapsed_time_ms = starter.elapsed_time(ender) / args.num_iterations
        avg_time_ms = elapsed_time_ms
        min_time_ms = elapsed_time_ms
        max_time_ms = elapsed_time_ms
        std_time_ms = elapsed_time_ms
        
        print(f"\nBenchmark Results:")
        print(f"Average inference time: {avg_time_ms:.2f} milliseconds")
        print(f"Min inference time: {avg_time_ms:.2f} milliseconds")
        print(f"Max inference time: {avg_time_ms:.2f} milliseconds")
        print(f"Standard deviation: N/A")
        print(f"Throughput: {1.0/avg_time_ms:.2f} FPS")
        
        return {
            "avg_time": avg_time_ms,
            "min_time": min_time_ms,
            "max_time": max_time_ms,
            "std_time": std_time_ms,
            "throughput": 1.0/avg_time_ms,
        }


def benchmark_policy_python_timer(model, arg_inputs, kwarg_inputs, args: argparse.Namespace = None):
    """
    Benchmark the policy inference performance.
    
    Args:
        policy: The Groot policy to benchmark
        step_data: The input data for inference
        num_iterations: Number of iterations to run for benchmarking
        warmup_iterations: Number of warmup iterations before benchmarking
    
    Returns:
        Dict containing benchmark results
    """
    
    with torch.inference_mode(), torch.no_grad():
        print(f"Running warmup for {args.warmup_iterations} iterations...")
        # Warmup iterations
        for i in range(args.warmup_iterations):
            _ = model(*arg_inputs, **kwarg_inputs)
            torch.cuda.synchronize()
        
        print(f"Running benchmark for {args.num_iterations} iterations...")

        # Benchmark iterations
        times = []
        for i in range(args.num_iterations):
            start_time = time.time()
            model(*arg_inputs, **kwarg_inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time_ms = np.mean(times)
        min_time_ms = np.min(times)
        max_time_ms = np.max(times)
        std_time_ms = np.std(times)

        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # # Convert to milliseconds for reporting
        avg_time_ms = avg_time * 1000
        min_time_ms = min_time * 1000
        max_time_ms = max_time * 1000
        std_time_ms = std_time * 1000
        
        print(f"\nBenchmark Results:")
        print(f"Average inference time: {avg_time_ms:.2f} milliseconds")
        print(f"Min inference time: {min_time_ms:.2f} milliseconds")
        print(f"Max inference time: {max_time_ms:.2f} milliseconds")
        print(f"Standard deviation: {std_time_ms:.2f} milliseconds")
        print(f"Throughput: {1.0/avg_time_ms:.2f} FPS")
        
        return {
            "avg_time": avg_time_ms,
            "min_time": min_time_ms,
            "max_time": max_time_ms,
            "std_time": std_time_ms,
            "throughput": 1.0/avg_time_ms,
        }

def benchmark_policy(model, arg_inputs, kwarg_inputs, args: argparse.Namespace = None):
    """
    Benchmark the policy inference performance.
    """
    if args.benchmark == "cuda_event":
        return benchmark_policy_cuda_event(model, arg_inputs, kwarg_inputs, args)
    elif args.benchmark == "python_timer":
        return benchmark_policy_python_timer(model, arg_inputs, kwarg_inputs, args)


def compare_benchmark_outputs(pyt_timings, trt_timings):
    """
    Compare the benchmark outputs between PyTorch and TensorRT models.
    
    Args:
        pyt_timings: Dictionary containing PyTorch benchmark results
        trt_timings: Dictionary containing TensorRT benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON (PyTorch vs TensorRT)")
    print("="*60)
    
    # Calculate speedup metrics
    speedup = pyt_timings["avg_time"] / trt_timings["avg_time"]
    throughput_improvement = (trt_timings["throughput"] - pyt_timings["throughput"]) / pyt_timings["throughput"] * 100
    
    # Format and display comparison
    max_label_width = 35
    
    print(f'{"Average Inference Time:".ljust(max_label_width)} PyTorch: {pyt_timings["avg_time"]:.4f} ms | TensorRT: {trt_timings["avg_time"]:.4f} ms')
    print(f'{"Min Inference Time:".ljust(max_label_width)} PyTorch: {pyt_timings["min_time"]:.4f} ms | TensorRT: {trt_timings["min_time"]:.4f} ms')
    print(f'{"Max Inference Time:".ljust(max_label_width)} PyTorch: {pyt_timings["max_time"]:.4f} ms | TensorRT: {trt_timings["max_time"]:.4f} ms')
    print(f'{"Standard Deviation:".ljust(max_label_width)} PyTorch: {pyt_timings["std_time"]:.4f} ms | TensorRT: {trt_timings["std_time"]:.4f} ms')
    print(f'{"Throughput (FPS):".ljust(max_label_width)} PyTorch: {pyt_timings["throughput"]:.2f} | TensorRT: {trt_timings["throughput"]:.2f}')
    
    print("\n" + "-"*60)
    print("PERFORMANCE IMPROVEMENTS")
    print("-"*60)
    print(f'{"Speedup (x faster):".ljust(max_label_width)} {speedup:.2f}x')
    print(f'{"Throughput Improvement:".ljust(max_label_width)} {throughput_improvement:.1f}%')
    
    if speedup > 1.0:
        print(f'{"Result:".ljust(max_label_width)} TensorRT is {speedup:.2f}x faster than PyTorch')
    else:
        print(f'{"Result:".ljust(max_label_width)} PyTorch is {1/speedup:.2f}x faster than TensorRT')
    
    print("="*60)