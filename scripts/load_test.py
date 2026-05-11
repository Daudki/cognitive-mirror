#!/usr/bin/env python3
"""Simple load testing script for the Cognitive Mirror API."""

import time
import json
import urllib.request
import concurrent.futures
from typing import List, Dict


def make_prediction(text: str, url: str = "http://localhost:5000/api/v1/predict") -> Dict:
    """Make a single prediction request."""
    data = json.dumps({"text": text}).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode())
    
    elapsed = (time.perf_counter() - start) * 1000
    result["_client_time_ms"] = round(elapsed, 2)
    return result


def run_load_test(num_requests: int = 100, concurrency: int = 5):
    """Run a concurrent load test."""
    test_texts = [
        "I am feeling very happy today!",
        "This is making me quite anxious and worried.",
        "I'm not sure how to feel about this situation.",
        "What a wonderful surprise!",
        "I am so frustrated with everything right now.",
    ]
    
    print(f"Starting load test: {num_requests} requests, {concurrency} concurrent")
    print("-" * 60)
    
    start_time = time.time()
    successful = 0
    failed = 0
    times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            text = test_texts[i % len(test_texts)]
            futures.append(executor.submit(make_prediction, text))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                successful += 1
                times.append(result.get("processing_time_ms", 0))
            except Exception as e:
                failed += 1
                print(f"  Error: {e}")
    
    total_time = time.time() - start_time
    
    # Statistics
    print(f"\nResults:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Successful:     {successful}")
    print(f"  Failed:         {failed}")
    print(f"  Throughput:     {successful / total_time:.1f} req/s")
    
    if times:
        times.sort()
        print(f"  Server latency (mean): {sum(times) / len(times):.1f}ms")
        print(f"  Server latency (p50):  {times[len(times)//2]:.1f}ms")
        print(f"  Server latency (p95):  {times[int(len(times)*0.95)]:.1f}ms")
        print(f"  Server latency (p99):  {times[int(len(times)*0.99)]:.1f}ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test Cognitive Mirror API")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("-c", "--concurrency", type=int, default=5, help="Concurrent workers")
    
    args = parser.parse_args()
    run_load_test(args.requests, args.concurrency)
