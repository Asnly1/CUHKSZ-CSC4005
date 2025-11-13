import matplotlib.pyplot as plt

cores = [1, 4, 8, 16, 32]

std_sort = [10077, 5031, 2537, 1288, 696]
merge_sort = [14636, 8136, 3871, 1889, 1111]
quick_sort = [12579, 7302, 4189, 2930, 2190]
radix_sort = [496, 496, 496, 496, 496]

plt.figure(figsize=(10,6))

plt.plot(cores, std_sort, marker='o', label='std::sort')
plt.plot(cores, merge_sort, marker='o', label='Merge Sort')
plt.plot(cores, quick_sort, marker='o', label='Quick Sort')
plt.plot(cores, radix_sort, marker='o', label='Radix Sort')

plt.xlabel('Number of Cores')
plt.ylabel('Runtime')
plt.title('Sorting Performance vs Number of Cores')
plt.legend()
plt.grid(True)

plt.show()
