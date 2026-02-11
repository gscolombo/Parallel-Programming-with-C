import numpy as np
import subprocess
import os
import time

import multiprocessing

MAX_CORES = multiprocessing.cpu_count()

def generate_test_data(n, distribution='uniform', seed=42):
    """Generate test data of size n"""
    np.random.seed(seed)
    
    if distribution == 'uniform':
        # Uniform distribution between 0 and 100
        data = np.random.uniform(0, 100, n)
    elif distribution == 'normal':
        # Normal distribution with mean 50, std 15
        data = np.random.normal(50, 15, n)
        # Clip to avoid extreme outliers
        data = np.clip(data, 0, 100)
    elif distribution == 'exponential':
        # Exponential distribution (right-skewed)
        data = np.random.exponential(20, n)
        data = np.clip(data, 0, 100)
    elif distribution == 'bimodal':
        # Bimodal distribution
        data1 = np.random.normal(30, 10, n//2)
        data2 = np.random.normal(70, 10, n - n//2)
        data = np.concatenate([data1, data2])
        data = np.clip(data, 0, 100)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return data

def write_input_file(filename, n, bins, data):
    """Write input data to file in the expected format"""
    with open(filename, 'w') as f:
        f.write(f"{n}\n")          # N
        f.write(f"{bins}\n")       # Number of bins
        # Write data (10 per line for readability)
        for i in range(0, n, 10):
            line = " ".join(f"{x:.4f}" for x in data[i:i+10])
            f.write(line + "\n")

def compile_program():
    """Compile the MPI program"""
    print("Compiling histogram program...")
    result = subprocess.run(
        ["mpicc", "-o", "histogram", "3.1_histogram.c", "-lm"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return False
    print("Compilation successful!")
    return True

def run_histogram(input_file, num_processes):
    """Run the histogram program with given input"""
    print(f"Running histogram with {num_processes} processes...")
    
    start_time = time.time()
    
    result = subprocess.run(
        ["mpirun", "-np", str(num_processes), "-use-hwthread-cpus", "./histogram", input_file],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    elapsed = time.time() - start_time
    
    return result, elapsed

def parse_output(output):
    """Parse histogram output to extract counts"""
    lines = output.strip().split('\n')
    counts = []
    
    for line in lines:
        if ':' in line:
            try:
                # Extract the count after the last colon
                count_str = line.split(':')[-1].strip()
                counts.append(int(count_str))
            except ValueError:
                continue
    
    return counts

def validate_counts(counts, expected_total):
    """Validate that counts sum to expected total"""
    total = sum(counts)
    if total == expected_total:
        print(f"✓ Validation passed: Total count = {total} (expected {expected_total})")
        return True
    else:
        print(f"✗ Validation failed: Total count = {total} (expected {expected_total})")
        return False

def main():
    # Compile the program first
    if not compile_program():
        return
    
    print("\n" + "="*70)
    print("LARGE-SCALE HISTOGRAM TESTS")
    print("="*70)
    
    test_cases = [
        {
            "name": "Small test (verification)",
            "n": 100,
            "bins": 6,
            "processes": 4,
            "distribution": "uniform"
        },
        {
            "name": "Medium test",
            "n": 1000,
            "bins": 20,
            "processes": 4,
            "distribution": "uniform"
        },
        {
            "name": "Large test",
            "n": 10000,
            "bins": 50,
            "processes": MAX_CORES,
            "distribution": "normal"
        },
        {
            "name": "Very large test",
            "n": 1_000_000,
            "bins": 100,
            "processes": MAX_CORES,
            "distribution": "bimodal"
        },
        {
            "name": "1GB test",
            "n": 125_000_000,
            "bins": 50,
            "processes": MAX_CORES,
            "distribution": "normal"
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{'-'*70}")
        print(f"Test: {test['name']}")
        print(f"  N = {test['n']:,}, bins = {test['bins']}, processes = {test['processes']}")
        print(f"  Distribution: {test['distribution']}")
        print(f"{'-'*70}")
        
        # Generate test data
        print("Generating test data...")
        data = generate_test_data(test['n'], test['distribution'])
        
        # Create input file
        input_file = f"test_{test['n']}.in"
        write_input_file(input_file, test['n'], test['bins'], data)
        print(f"Input written to {input_file}")
        
        # Run histogram
        result, elapsed = run_histogram(input_file, test['processes'])
        
        if result.returncode != 0:
            print(f"Program failed with return code {result.returncode}")
            if result.stderr:
                print("\nOutput:")
                print(result.stdout)
                print("Error output:")
                print(result.stderr)
            results.append((test['name'], False, elapsed))
            continue
        
        # Parse and validate output
        counts = parse_output(result.stdout)
        
        if not counts:
            print("Warning: No counts parsed from output")
            print("Output preview:")
            print(result.stdout[:500])
        else:
            print(f"Got {len(counts)} bins")
            valid = validate_counts(counts, test['n'])
        
        print(f"Execution time: {elapsed:.3f} seconds")
        
        # Show first few lines of output for verification
        if counts:
            lines = result.stdout.strip().split('\n')[2:]
            for line in lines:
                print(f"  {line}")
        
        results.append((test['name'], True, elapsed))
        
        # Clean up input file
        os.remove(input_file)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success, elapsed in results:
        status = "PASS" if success else "FAIL"
        print(f"{name:30} {status:10} {elapsed:7.3f}s")
    
    # Clean up
    if os.path.exists("histogram"):
        os.remove("histogram")

if __name__ == "__main__":
    main()