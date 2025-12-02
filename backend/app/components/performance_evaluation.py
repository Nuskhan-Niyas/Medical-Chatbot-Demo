# performance_evaluation.py

import time
import csv
from app.components.retriever import create_qa_chain

# -----------------------------
# Configuration
# -----------------------------
TEST_QUERIES = [
    "What are the symptoms of diabetes?",
    "What is hypertension?",
    "What are the causes of asthma?",
    "How is tuberculosis treated?",
    "What are common side effects of insulin?",
]

NUM_REPEAT = 10  # number of times to repeat each query for throughput measurement
CSV_FILE = "performance_results.csv"

# -----------------------------
# Initialize QA chain
# -----------------------------
print("Initializing QA chain...")
qa_chain = create_qa_chain()
if qa_chain is None:
    raise RuntimeError("QA chain could not be created")

print("✅ QA chain ready\n")

# -----------------------------
# Measure latency for each query
# -----------------------------
latency_results = []

for query in TEST_QUERIES:
    # Warm-up run
    _ = qa_chain.invoke({"query": query})["result"]
    
    start_time = time.time()
    result = qa_chain.invoke({"query": query})["result"]
    end_time = time.time()
    
    latency = end_time - start_time
    latency_results.append({"query": query, "latency_sec": latency})
    print(f"Query: {query}")
    print(f"Result: {result}")
    print(f"Latency: {latency:.3f} sec\n")

# Average latency
avg_latency = sum(r["latency_sec"] for r in latency_results) / len(latency_results)
print(f"Average Latency: {avg_latency:.3f} sec\n")

# -----------------------------
# Measure throughput
# -----------------------------
num_queries = len(TEST_QUERIES) * NUM_REPEAT
start_time = time.time()

for _ in range(NUM_REPEAT):
    for query in TEST_QUERIES:
        _ = qa_chain.invoke({"query": query})["result"]

end_time = time.time()
total_time = end_time - start_time
throughput_per_sec = num_queries / total_time
throughput_per_min = throughput_per_sec * 60

print(f"Total Queries: {num_queries}")
print(f"Total Time: {total_time:.2f} sec")
print(f"Throughput: {throughput_per_sec:.2f} req/sec, {throughput_per_min:.2f} req/min\n")

# -----------------------------
# Save results to CSV
# -----------------------------
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "latency_sec"])
    writer.writeheader()
    writer.writerows(latency_results)
    writer.writerow({"query": "Average", "latency_sec": avg_latency})

print(f"✅ Latency results saved to {CSV_FILE}")
print("Performance evaluation completed.")
