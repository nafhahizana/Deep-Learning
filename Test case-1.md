code:

webcam_data = [
    {"Image": "Live Face 1", "Expected": "John", "Recognized": "John"},
    {"Image": "Live Face 2", "Expected": "Alice", "Recognized": "John"},
    {"Image": "Unknown", "Expected": "", "Recognized": "Unknown"}
]

# Evaluate correctness
for entry in webcam_data:
    expected = entry["Expected"]
    recognized = entry["Recognized"]
    correct = "Y" if expected == recognized else "N"
    entry["Correct"] = correct

# Display results
print(f"{'Webcam Image':<15} {'Expected Name':<15} {'Recognized Name':<17} {'Correct (Y/N)':<13}")
for entry in webcam_data:
    print(f"{entry['Image']:<15} {entry['Expected']:<15} {entry['Recognized']:<17} {entry['Correct']:<13}")   

output:

![Screenshot_3-9-2025_102850_colab research google com](https://github.com/user-attachments/assets/846c3270-b372-40f4-b488-fdde763f3af4)
