# test_learn.py
from llm import get_llm
from relevance import build_relevance_chain
from generate import generate, state
from learn import learn, looks_like_solution

relevance_chain = build_relevance_chain(get_llm())
llm = get_llm()

FENCE = "`" * 3  # three backticks, built here so nothing breaks when copying


def ask(q):
    print(f"\n=== {q}")
    for t in generate(llm, relevance_chain, q):
        print(t, end="", flush=True)
    print()


# 1. Unknown question -> should say "couldn't find relevant knowledge"
ask("Train a CNN in PyTorch")

solution = (
    "Here is the correct implementation:\n"
    f"{FENCE}python\n"
    "import torch.nn as nn\n\n"
    "class SimpleCNN(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(1, 16, 3)\n"
    "        self.fc = nn.Linear(16 * 26 * 26, 10)\n\n"
    "    def forward(self, x):\n"
    "        x = self.conv(x).relu().flatten(1)\n"
    "        return self.fc(x)\n"
    f"{FENCE}"
)

# 2. User supplies a solution -> store it in Chroma
if state["pending_query"] and looks_like_solution(solution):
    print(learn(solution))

# 3. Same question again -> should now generate code using the stored solution
ask("Train a CNN in PyTorch")