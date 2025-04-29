def multiply_nested_list(lst, factor):
    result = []
    for item in lst:
        if isinstance(item, list):
            # Recursively handle sublists
            result.append(multiply_nested_list(item, factor))
        else:
            # Multiply number
            result.append(item * factor)
    return result




nested = [1, [2, [3, 4], 5], [1, [9, 98, 12]], 6]
factor = 2
output = multiply_nested_list(nested, factor)
print(output)  # Output: [2, [4, [6, 8], 10], 12]
