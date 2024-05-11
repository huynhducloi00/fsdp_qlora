import datasets
from datasets import load_dataset
squad_dataset = load_dataset('squad')
print(squad_dataset['train'][0])
def bubble_sort(arr):  
    n = len(arr)  
  
    # Traverse through all array elements  
    for i in range(n):  
        # Last i elements are already sorted, so we don't need to check them  
        for j in range(0, n-i-1):  
            # Swap if the element found is greater than the next element  
            if arr[j] > arr[j+1]:  
                arr[j], arr[j+1] = arr[j+1], arr[j]  
  
# Example usage  
if __name__ == "__main__":  
    # Sample list to be sorted  
    sample_list = [64, 34, 25, 12, 22, 11, 90]  
  
    # Display the original list  
    print("Original List:", sample_list)  
  
    # Apply Bubble Sort  
    bubble_sort(sample_list)  
  
    # Display the sorted list  
    print("Sorted List:", sample_list)  