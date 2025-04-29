def sum_all_nested_lists(list_of_lists):
    def recursive_sum(lists):
        if isinstance(lists[0], list):
            return [recursive_sum([lst[i] for lst in lists]) for i in range(len(lists[0]))]
        else:
            return sum(lists)
    
    return recursive_sum(list_of_lists)



lists = [
    [1254, [521, [346, 485], 21], 69],
    [220, [40, [80, 30], 10], 80],
    [100, [200, [300, 400], 500], 600]
]

result = sum_all_nested_lists(lists)
print(result)  # Output: [111, [222, [333, 444], 555], 666]
