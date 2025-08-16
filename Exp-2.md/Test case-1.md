code:

test_cases = [
    {"image": "Image of 7", "expected": 7, "model_output": 7},
    {"image": "Image of 3", "expected": 3, "model_output": 3},
    {"image": "Image of 8", "expected": 8, "model_output": 8},
    {"image": "Image of 1", "expected": 1, "model_output": 2},
]

for case in test_cases:
    correct = "Y" if case["expected"] == case["model_output"] else "N"
    print(f"{case['image']:12} | Expected: {case['expected']} | Model Output: {case['model_output']} | Correct: {correct}")

output:

![Deep Learning exp2 test case 12025-08-16 at 12 56 29_5167f295](https://github.com/user-attachments/assets/121ce8a5-f3fa-4661-9c55-a6db400249c5)
