# Actual completed runs

| method | seed | model | rank | targets | fraction | steps | test_loss | test_accuracy | test_macro_f1 | trainable_parameters | training_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 7 | HuggingFaceTB/SmolLM2-135M | — | n/a | 1 | 0 | 9.2666 | 0.125 | 0.11111 | 0 | 0 |
| dora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.3733 | 0.125 | 0.11111 | 506880 | 25.14 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 0.98868 | 0.875 | 0.46667 | 460800 | 12.518 |
| pissa | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 1.1013 | 0.875 | 0.46667 | 460800 | 12.935 |
| baseline | 7 | HuggingFaceTB/SmolLM2-135M | — | n/a | 1 | 0 | 9.2666 | 0.125 | 0.11111 | 0 | 0 |
| dora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.3743 | 0.125 | 0.11111 | 506880 | 17.996 |
| full | 7 | HuggingFaceTB/SmolLM2-135M | — | n/a | 1 | 12 | 0.36963 | 0.82812 | 0.58466 | 134515008 | 56.419 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.6169 | 0.125 | 0.11111 | 460800 | 32.272 |
| pissa | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 0.99723 | 0.875 | 0.46667 | 460800 | 17.324 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | combined | 1 | 12 | 5.3495 | 0.125 | 0.11111 | 1221120 | 22.299 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | ffn | 1 | 12 | 6.9722 | 0.125 | 0.11111 | 760320 | 18.576 |
| baseline | 7 | HuggingFaceTB/SmolLM2-135M | — | n/a | 1 | 0 | 9.2666 | 0.125 | 0.11111 | 0 | 0 |
| dora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.3743 | 0.125 | 0.11111 | 506880 | 21.054 |
| full | 7 | HuggingFaceTB/SmolLM2-135M | — | n/a | 1 | 12 | 0.36963 | 0.82812 | 0.58466 | 134515008 | 86.45 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.6169 | 0.125 | 0.11111 | 460800 | 30.024 |
| pissa | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 0.99723 | 0.875 | 0.46667 | 460800 | 14.399 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 16 | attention | 1 | 12 | 7.5069 | 0.125 | 0.11111 | 1843200 | 12.893 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 32 | attention | 1 | 12 | 7.4706 | 0.125 | 0.11111 | 3686400 | 17.285 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 8 | attention | 1 | 12 | 7.4873 | 0.125 | 0.11111 | 921600 | 11.851 |
| lora | 17 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.5713 | 0.125 | 0.11111 | 460800 | 26.93 |
| lora | 27 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 7.5098 | 0.125 | 0.11111 | 460800 | 22.966 |
| pissa | 17 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 1.0293 | 0.875 | 0.46667 | 460800 | 15.064 |
| pissa | 27 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 1 | 12 | 0.94473 | 0.875 | 0.46667 | 460800 | 15.114 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 0.1 | 12 | 7.6128 | 0.125 | 0.11111 | 460800 | 19.011 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 0.25 | 12 | 7.618 | 0.125 | 0.11111 | 460800 | 18.246 |
| lora | 7 | HuggingFaceTB/SmolLM2-135M | 4 | attention | 0.5 | 12 | 7.6804 | 0.125 | 0.11111 | 460800 | 12.014 |
| full | 7 | random-tiny-transformer | — | n/a | 1 | 60 | 2.1718 | — | — | 62784 | 1.3815 |
| baseline | 7 | random-tiny-transformer | — | n/a | 1 | 0 | 3.6385 | — | — | 0 | 0 |
| dora | 7 | random-tiny-transformer | 4 | attention | 1 | 60 | 3.5426 | — | — | 3456 | 0.99899 |
| full | 7 | random-tiny-transformer | — | n/a | 1 | 60 | 2.1718 | — | — | 62784 | 1.1786 |
| lora | 7 | random-tiny-transformer | 4 | attention | 1 | 60 | 3.5446 | — | — | 3072 | 0.92862 |
| pissa | 7 | random-tiny-transformer | 4 | attention | 1 | 60 | 3.4426 | — | — | 3072 | 0.91559 |

Do not pool different data hashes, initialization modes, or model revisions. Target perplexity is conditional on labels for SMS, not whole-text perplexity.
