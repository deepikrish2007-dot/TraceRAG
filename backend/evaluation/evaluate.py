
import json
from pathlib import Path

from backend.src.answer import retrieve_passages
from backend.evaluation.metrics import evaluate_retrieval


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "test_dataset.json"

TOP_K = 5


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_test_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "test_dataset.json must contain a JSON list."
        )

    return dataset


# --------------------------------------------------
# DISPLAY RETRIEVED PASSAGES
# --------------------------------------------------

def display_passages(retrieved):

    print("\nRetrieved Passages:")
    print("-" * 60)

    if not retrieved:
        print("No passages retrieved.")
        return

    for rank, passage in enumerate(
        retrieved[:TOP_K],
        start=1
    ):

        source = passage.get("source", "Unknown")
        page = passage.get("page", "Unknown")
        distance = passage.get("distance", "N/A")
        text = passage.get("text", "")

        print(f"\nPassage {rank}")
        print(f"Source   : {source}")
        print(f"Page     : {page}")
        print(f"Distance : {distance}")
        print(f"Text     : {text[:250]}...")


# --------------------------------------------------
# EVALUATE ONE TEST CASE
# --------------------------------------------------

def evaluate_single_test(test, test_number):

    test_id = test.get(
        "id",
        f"T{test_number}"
    )

    question = test["question"]

    expected_evidence = test.get(
        "expected_evidence",
        []
    )

    print("\n" + "=" * 70)
    print(f"TEST CASE {test_number} - {test_id}")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    # Retrieve passages from TraceRAG
    retrieved = retrieve_passages(
        question,
        top_k=TOP_K
    )

    if retrieved is None:
        retrieved = []

    display_passages(retrieved)

    # Calculate evaluation metrics
    metrics = evaluate_retrieval(
        expected_evidence=expected_evidence,
        retrieved=retrieved,
        k=TOP_K
    )

    # Display metrics
    print("\nEvaluation Metrics:")
    print("-" * 45)

    print(
        f"Precision@{TOP_K:<2}       : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall@{TOP_K:<2}          : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"MRR                  : "
        f"{metrics['mrr']:.4f}"
    )

    print(
        f"Hit Rate@{TOP_K:<2}        : "
        f"{metrics['hit_rate']:.4f}"
    )

    print(
        f"nDCG@{TOP_K:<2}           : "
        f"{metrics['ndcg']:.4f}"
    )

    print(
        f"Source Coverage      : "
        f"{metrics['source_coverage']:.4f}"
    )

    print(
        f"Relevant Pages       : "
        f"{metrics['relevant_count']}"
    )

    print(
        f"Retrieved Passages   : "
        f"{metrics['retrieved_count']}"
    )

    print(
        f"Expected Evidence    : "
        f"{metrics['expected_evidence_count']}"
    )

    print(
        f"Evidence Retrieved   : "
        f"{metrics['covered_evidence_count']}"
    )

    return metrics


# --------------------------------------------------
# CALCULATE AVERAGE METRICS
# --------------------------------------------------

def calculate_average_metrics(all_metrics):

    metric_names = [
        "precision",
        "recall",
        "mrr",
        "hit_rate",
        "ndcg",
        "source_coverage"
    ]

    averages = {}

    for name in metric_names:

        averages[name] = sum(
            item[name]
            for item in all_metrics
        ) / len(all_metrics)

    return averages


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

def display_final_report(
    all_metrics,
    failed_tests
):

    averages = calculate_average_metrics(
        all_metrics
    )

    print("\n\n")
    print("=" * 70)
    print("                 TRACERAG FINAL REPORT")
    print("=" * 70)

    print(
        f"\nSuccessfully evaluated: {len(all_metrics)}"
    )

    print(
        f"Failed test cases      : {failed_tests}"
    )

    print("\nAverage Retrieval Metrics:")
    print("-" * 55)

    print(
        f"Average Precision@{TOP_K}: "
        f"{averages['precision']:.4f}"
    )

    print(
        f"Average Recall@{TOP_K}   : "
        f"{averages['recall']:.4f}"
    )

    print(
        f"Mean Reciprocal Rank    : "
        f"{averages['mrr']:.4f}"
    )

    print(
        f"Average Hit Rate@{TOP_K}: "
        f"{averages['hit_rate']:.4f}"
    )

    print(
        f"Average nDCG@{TOP_K}    : "
        f"{averages['ndcg']:.4f}"
    )

    print(
        f"Average Source Coverage: "
        f"{averages['source_coverage']:.4f}"
    )

    print("\n" + "=" * 70)
    print("              EVALUATION COMPLETED")
    print("=" * 70)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("                  TRACERAG EVALUATION")
    print("=" * 70)

    # Load evaluation dataset
    dataset = load_test_dataset()

    print(f"\nTotal test cases: {len(dataset)}")
    print(f"Top K value     : {TOP_K}")

    all_metrics = []
    failed_tests = 0

    # Run all test cases
    for index, test in enumerate(
        dataset,
        start=1
    ):

        try:

            metrics = evaluate_single_test(
                test,
                index
            )

            all_metrics.append(metrics)

        except Exception as error:

            failed_tests += 1

            print(
                f"\nError in test case {index}: {error}"
            )

    # Generate final report
    if all_metrics:

        display_final_report(
            all_metrics,
            failed_tests
        )

    else:

        print("\nNo test cases were successfully evaluated.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
