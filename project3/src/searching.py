import matplotlib.pyplot as plt

cores = [1, 4, 8, 16, 32]

lower_bound = [1880, 1880, 1880, 1880, 1880]
search_cpu = [580, 234, 145, 98, 96]
search_gpu = [364, 364, 364, 364, 364]

plt.figure(figsize=(10,6))

plt.plot(cores, lower_bound, marker='o', label='std::lower_bound')
plt.plot(cores, search_cpu, marker='o', label='Search (CPU)')
plt.plot(cores, search_gpu, marker='o', label='Search (GPU)')

plt.xlabel('Number of Cores')
plt.ylabel('Runtime')
plt.title('Search Performance vs Number of Cores')
plt.legend()
plt.grid(True)

plt.show()
