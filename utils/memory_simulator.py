import time
import argparse
from numba import cuda
import numpy as np
import sys
import traceback

def simulate_memory_fluctuation(min_gb, max_gb):
    """
    Dynamically allocates and deallocates GPU memory to fluctuate
    between a specified min and max gigabyte range.

    Args:
        min_gb (float): The lower bound of memory usage in Gigabytes.
        max_gb (float): The upper bound of memory usage in Gigabytes.
    """
    try:
        cuda.select_device(0)
    except IndexError:
        print("Error: No NVIDIA CUDA-enabled GPU found or selected.")
        return

    allocated_buffers = []

    try:
        device = cuda.get_current_device()
        print(f"--- GPU Memory Fluctuation Simulator ---")
        print(f"Found GPU: {device.name.decode('UTF-8')}")

        # **FIX:** Use cuda.mem_get_info() to get total memory.
        # This is the correct and robust way to query memory.
        device = cuda.get_current_device()
        ctx = cuda.current_context()
        free_mem_bytes, total_mem_bytes = ctx.get_memory_info()
        total_mem_gb = total_mem_bytes / (1024**3)
        print(f"Total GPU Memory: {total_mem_gb:.2f} GB")

        # --- Validate Inputs ---
        if min_gb >= max_gb:
            print("Error: Minimum memory must be less than maximum memory.")
            return
        if max_gb > total_mem_gb * 0.95: # Safety margin
            print(f"Warning: Max memory {max_gb:.2f} GB is very close to total GPU memory.")
            print("Reducing max to 95% of total to avoid system instability.")
            max_gb = total_mem_gb * 0.95

        # --- Convert GB to bytes for allocation ---
        min_bytes = int(min_gb * (1024**3))
        max_bytes = int(max_gb * (1024**3))

        chunk_size_bytes = int(128 * (1024**2)) # 128 MB chunks
        current_bytes_allocated = 0

        # Start by allocating up to the minimum
        print("Allocating initial minimum memory...")
        while current_bytes_allocated < min_bytes:
            gpu_buffer = cuda.device_array(shape=(chunk_size_bytes,), dtype=np.uint8)
            allocated_buffers.append(gpu_buffer)
            current_bytes_allocated += chunk_size_bytes

        print(f"Simulating memory usage between {min_gb:.2f} GB and {max_gb:.2f} GB.")
        print("Press Ctrl+C to exit and release all memory.")

        # Main loop to increase and decrease memory
        while True:
            # --- Phase 1: Increase memory to MAX ---
            print("\nIncreasing memory usage to max...")
            while current_bytes_allocated < max_bytes:
                try:
                    gpu_buffer = cuda.device_array(shape=(chunk_size_bytes,), dtype=np.uint8)
                    allocated_buffers.append(gpu_buffer)
                    current_bytes_allocated += chunk_size_bytes
                    print(f"\rCurrent allocated GPU memory: {current_bytes_allocated / (1024**3):.2f} GB", end="")
                    time.sleep(0.1)
                except cuda.CudaAPIError as e:
                    print(f"\nCould not allocate more memory: {e}")
                    break

            print(f"\rCurrent allocated GPU memory: {current_bytes_allocated / (1024**3):.2f} GB - Reached MAX")
            time.sleep(2)

            # --- Phase 2: Decrease memory to MIN ---
            print("\nDecreasing memory usage to min...")
            while current_bytes_allocated > min_bytes and len(allocated_buffers) > 0:
                allocated_buffers.pop()
                current_bytes_allocated -= chunk_size_bytes
                if current_bytes_allocated < 0: current_bytes_allocated = 0
                print(f"\rCurrent allocated GPU memory: {current_bytes_allocated / (1024**3):.2f} GB", end="")
                time.sleep(0.1)

            print(f"\rCurrent allocated GPU memory: {current_bytes_allocated / (1024**3):.2f} GB - Reached MIN")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nExiting... Releasing all GPU memory.")

    except Exception as e:
        print("\nAn unexpected error occurred:")
        traceback.print_exc()

    finally:
        allocated_buffers.clear()
        try:
            cuda.close()
            print("GPU context closed and all memory has been released.")
        except Exception:
            print("All GPU memory should be released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamically simulate GPU memory usage by fluctuating between a min and max value.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--min-gb',
        type=float,
        required=True,
        help="The minimum GPU memory to use, in Gigabytes (e.g., 1.5)."
    )
    parser.add_argument(
        '--max-gb',
        type=float,
        required=True,
        help="The maximum GPU memory to use, in Gigabytes (e.g., 4)."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    simulate_memory_fluctuation(args.min_gb, args.max_gb)
