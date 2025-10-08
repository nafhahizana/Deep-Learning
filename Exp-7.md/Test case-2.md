code:

import pandas as pd

test_cases = [
    {
        "Input": "He is reading a book",
        "Normal Output": "वह एक डकताब पढ़ रहा है।",
        "Attention Output": "वह एक पुस्तक पढ़ रहा है।"
    },
    {
        "Input": "She is cooking food",
        "Normal Output": "वह खाना बना रही है।",
        "Attention Output": "वह भोजन पका रही है।"
    },
    {
        "Input": "They are playing outside",
        "Normal Output": "वे बाहर खेल रहे हैं।",
        "Attention Output": "वे बाहर खेल रहे हैं।"
    }
]

df = pd.DataFrame(test_cases)
pd.set_option('display.max_colwidth', None)

print(df.to_string(index=False))

output:

<img width="544" height="76" alt="Screenshot 2025-10-08 115629" src="https://github.com/user-attachments/assets/f5ac5022-acc1-47d9-9292-4bfdd4e1b3f3" />
