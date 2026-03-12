import os
import warnings

warnings.filterwarnings("ignore")

from datasets import Dataset
from google import genai
from ragas import evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory(
    "gemini-2.0-flash-lite-001", provider="google", client=client
)
embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [
        ["France is a country in Western Europe. Paris is its capital."]
    ],
    "ground_truth": ["Paris"],
}

dataset = Dataset.from_dict(data)

metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=embeddings),
]

results = evaluate(dataset, metrics=metrics)
print(results)
