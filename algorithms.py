import random
import time
import matplotlib.pyplot as plt

# Function to perform insertion sort
def insertion_sort(arr):
    # Counter for number of comparisons
    comparisons = 0  
    # Counter for number of swaps
    swaps = 0  
    # Start time for measuring execution time
    start_time = time.time()
    # Loop through the array
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Compare and swap elements
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
            comparisons += 1
            swaps += 1
        arr[j + 1] = key
        swaps += 1
    # End time for measuring execution time
    end_time = time.time()
    # Calculate execution time in milliseconds
    execution_time = (end_time - start_time) * 1000  
    return comparisons, swaps, execution_time

# Function to perform merge sort
def merge_sort(arr):
    # Helper function to merge two sorted arrays
    def merge(left, right):
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    # Recursive function to perform merge sort
    def sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = sort(arr[:mid])
        right = sort(arr[mid:])
        return merge(left, right)

    # Start time for measuring execution time
    start_time = time.time()
    sorted_arr = sort(arr)
    # End time for measuring execution time
    end_time = time.time()
    # Number of comparisons for merge sort is always n - 1
    comparisons = len(arr) - 1
    # Calculate execution time in milliseconds
    execution_time = (end_time - start_time) * 1000

    return comparisons, 0, execution_time

# Function to perform quick sort
def quick_sort(arr):
    # Helper function to partition the array
    def partition(arr, low, high):
        pivot = arr[low]
        i = low + 1
        j = high
        while True:
            while i <= j and arr[i] <= pivot:
                i += 1
            while i <= j and arr[j] >= pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
            else:
                break
        arr[low], arr[j] = arr[j], arr[low]
        return j

    # Helper function for iterative quick sort
    def quick_sort_iterative(arr):
        stack = [(0, len(arr) - 1)]
        comparisons = 0
        swaps = 0
        while stack:
            low, high = stack.pop()
            if low < high:
                pivot_index = partition(arr, low, high)
                stack.append((low, pivot_index - 1))
                stack.append((pivot_index + 1, high))
                comparisons += (high - low)
                swaps += (pivot_index - low)
        return comparisons, swaps

    comparisons, swaps = quick_sort_iterative(arr)
    return comparisons, swaps, 0

# Function to perform max heapify operation
def max_heapify(arr, n, i):
    comparisons = 0
    swaps = 0
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    # Compare with left child
    if l < n and arr[l] > arr[largest]:
        largest = l
    comparisons += 1

    # Compare with right child
    if r < n and arr[r] > arr[largest]:
        largest = r
    comparisons += 1

    # Swap if necessary and recursively max heapify
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        swaps += 1
        max_heapify(arr, n, largest)
    return comparisons, swaps

# Function to build a max heap from an array
def build_max_heap(arr):
    comparisons = 0
    swaps = 0
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        c, s = max_heapify(arr, n, i)
        comparisons += c
        swaps += s
    return comparisons, swaps

# Function to find the median using max heap
def find_median_max_heap(arr):
    comparisons, swaps = build_max_heap(arr)
    n = len(arr)
    for _ in range(n // 2):
        c, s = max_heapify(arr, n, 0)
        comparisons += c
        swaps += s
        arr[0], arr[n - 1] = arr[n - 1], arr[0]
        n -= 1
    median = arr[0] if len(arr) % 2 != 0 else (arr[0] + arr[1]) / 2
    return comparisons, swaps, median

# Function to perform quick select algorithm
def quick_select(arr, k):
    # Helper function to partition the array
    def partition(arr, left, right):
        pivot = arr[left]
        i = left + 1
        j = right
        comparisons = 0

        while True:
            while i <= j and arr[i] <= pivot:
                i += 1
                comparisons += 1
            while i <= j and arr[j] >= pivot:
                j -= 1
                comparisons += 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
            else:
                break

        arr[left], arr[j] = arr[j], arr[left]

        return j, comparisons

    comparisons = 0
    left = 0
    right = len(arr) - 1

    while left <= right:
        pivot_index, comparisons_partition = partition(arr, left, right)

        if pivot_index == k:
            comparisons += comparisons_partition
            return arr[k], comparisons
        elif pivot_index < k:
            comparisons += comparisons_partition
            left = pivot_index + 1
        else:
            comparisons += comparisons_partition
            right = pivot_index - 1

    return None, comparisons

# Function to find the median using quick select algorithm
def quick_select_median(arr):
    start_time = time.time()
    median, comparisons = quick_select(arr, len(arr) // 2)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    return median, comparisons, execution_time

# Function to find the median using quick select with median-of-three pivot selection
def quick_select_median_of_three(arr):
    first = arr[0]
    middle = arr[len(arr) // 2]
    last = arr[-1]
    pivot = sorted([first, middle, last])[1]
    pivot_index = arr.index(pivot)
    arr[0], arr[pivot_index] = arr[pivot_index], arr[0]
    start_time = time.time()
    median, comparisons = quick_select(arr, len(arr) // 2)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    return median, comparisons, execution_time

# Main function
def main():
    # Input the size of the array
    size = int(input("Enter the size of the array (between 1-8192): "))

    # Generate a random array of the specified size
    min_value = 1
    max_value = 8192
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    print("Random array:", random_array)

    # List of algorithms to be tested
    algorithms = [
        ("Insertion Sort", insertion_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort (with first element as pivot)", quick_sort),
        ("Max Heap", find_median_max_heap),
        ("Quick Select (with first element as pivot)", quick_select_median),
        ("Quick Select (with median-of-three pivot selection)", quick_select_median_of_three)
    ]

    # Lists to store results for each algorithm
    comparisons_list = []
    swaps_list = []
    exec_time_list = []

    # Iterate through each algorithm and perform analysis
    for name, algorithm in algorithms:
        print(f"\n{name}:")
        if algorithm in [insertion_sort, merge_sort, quick_sort]:
            worst_case = sorted(random_array, reverse=True)
            best_case = sorted(random_array)
            average_case = random_array.copy()
            comparisons_worst, swaps_worst, exec_time_worst = algorithm(worst_case)
            comparisons_best, swaps_best, exec_time_best = algorithm(best_case)
            comparisons_avg, swaps_avg, exec_time_avg = algorithm(average_case)
            median = average_case[len(average_case) // 2] if len(average_case) % 2 != 0 else (average_case[len(average_case) // 2 - 1] + average_case[len(average_case) // 2]) / 2
            print("Worst Case - Comparisons:", comparisons_worst, "Swaps:", swaps_worst, "Execution Time (ms):", exec_time_worst)
            print("Best Case - Comparisons:", comparisons_best, "Swaps:", swaps_best, "Execution Time (ms):", exec_time_best)
            print("Average Case - Comparisons:", comparisons_avg, "Swaps:", swaps_avg, "Execution Time (ms):", exec_time_avg)
            print("Median:", median)
        elif algorithm == find_median_max_heap:
            comparisons_worst, swaps_worst, median_worst = algorithm(random_array)
            print("Worst Case - Comparisons:", comparisons_worst, "Swaps:", swaps_worst, "Median:", median_worst)
            # Add zero execution time for max heap
            exec_time_worst = 0
        elif algorithm in [quick_select_median, quick_select_median_of_three]:
            worst_case = sorted(random_array, reverse=True)
            best_case = sorted(random_array)
            average_case = random_array.copy()
            median_worst, comparisons_worst, exec_time_worst = algorithm(worst_case)
            median_best, comparisons_best, exec_time_best = algorithm(best_case)
            median_avg, comparisons_avg, exec_time_avg = algorithm(average_case)
            print("Worst Case - Median:", median_worst, "Comparisons:", comparisons_worst, "Execution Time (ms):", exec_time_worst)
            print("Best Case - Median:", median_best, "Comparisons:", comparisons_best, "Execution Time (ms):", exec_time_best)
            print("Average Case - Median:", median_avg, "Comparisons:", comparisons_avg, "Execution Time (ms):", exec_time_avg)

        # Append results to lists
        comparisons_list.append((comparisons_worst, comparisons_best, comparisons_avg))
        swaps_list.append((swaps_worst, swaps_best, swaps_avg))
        exec_time_list.append((exec_time_worst, exec_time_best, exec_time_avg))

    # Plotting results
    x = ["Worst Case", "Best Case", "Average Case"]

    # Plotting comparison counts
    plt.figure(figsize=(12, 6))
    for i, (name, _) in enumerate(algorithms):
        comp_worst, comp_best, comp_avg = comparisons_list[i]
        plt.bar([f"{name} - {case}" for case in x], [comp_worst, comp_best, comp_avg], label=name)
    plt.title("Comparison Counts")
    plt.xlabel("Case")
    plt.ylabel("Comparisons")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc='upper right')
    plt.show()

    # Plotting swap counts
    plt.figure(figsize=(12, 6))
    for i, (name, _) in enumerate(algorithms):
        swap_worst, swap_best, swap_avg = swaps_list[i]
        plt.bar([f"{name} - {case}" for case in x], [swap_worst, swap_best, swap_avg], label=name)
    plt.title("Swap Counts")
    plt.xlabel("Case")
    plt.ylabel("Swaps")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc='upper right')
    plt.show()

    # Plotting execution times
    plt.figure(figsize=(12, 6))
    for i, (name, _) in enumerate(algorithms):
        exec_time_worst, exec_time_best, exec_time_avg = exec_time_list[i]
        plt.bar([f"{name} - {case}" for case in x], [exec_time_worst, exec_time_best, exec_time_avg], label=name)
    plt.title("Execution Times")
    plt.xlabel("Case")
    plt.ylabel("Execution Time (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc='upper right')
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()
