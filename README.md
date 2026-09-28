# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

### Chain design

```mermaid
flowchart LR
    A[Receipt images] --> B[Encode images as data URLs]
    B --> C[LangChain prompt]
    C --> D[DeepSeek V4 Flash Vision]
    D --> E[JSON amounts per receipt]
    E --> F[Aggregate with Decimal]
    F --> G[One HKD amount for each question]
    G --> H[Provided runner writes results.csv]
```

For each receipt, the LangChain prompt sends the image to `deepseek-v4-flash-vision-exp` and requests the final payment after rounding, the discounted subtotal before rounding, and the positive total of discounts. The model processes receipts in small parallel batches; a receipt is retried up to twice if its response is not valid JSON. Python parses the returned amounts and uses `Decimal` to sum actual payments for the first question and each subtotal plus its discounts for the second question. This keeps currency arithmetic deterministic and returns one HKD amount for each required query.
