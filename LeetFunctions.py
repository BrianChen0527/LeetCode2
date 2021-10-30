#functions file

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def find_averages_of_subarrays(K, arr):
    sums = []
    sum, start_pos = 0.0, 0


    for i in range(len(arr)):
        sum += arr[i]
        if(i-start_pos+1 == K):
            sums.append(sum/5)
            start_pos+=1
            sum -= arr[start_pos]
    return sums