
import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = BASE_DIR / "vector_db"
TEST_FILE = BASE_DIR / "evaluation" / "test_questions.json"
RESULT_FILE = BASE_DIR / "evaluation" / "results.json"

TOP_K = 3


def load_vector_database():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        str(VECTOR_DB_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_db


def evaluate_retrieval():

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    vector_db = load_vector_database()

    total_positive = 0
    successful_retrievals = 0

    evaluation_results = []

    print("\n========== RAG RETRIEVAL EVALUATION ==========\n")

    for test in questions:

        question = test["question"]
        keywords = [
            word.lower()
            for word in test["expected_keywords"]
        ]

        should_have_evidence = test["should_have_evidence"]

        # Retrieve documents along with FAISS distance scores
        results = vector_db.similarity_search_with_score(
            question,
            k=TOP_K
        )

        retrieved_text = " ".join(
            doc.page_content.lower()
            for doc, score in results
        )

        matched_keywords = [
            word for word in keywords
            if word in retrieved_text
        ]

        if should_have_evidence:

            total_positive += 1

            passed = (
                len(keywords) > 0
                and len(matched_keywords) == len(keywords)
            )

            if passed:
                successful_retrievals += 1

            status = "PASS" if passed else "FAIL"

        else:
            status = "MANUAL REVIEW"

        print(f"\nQuestion {test['id']}: {question}")

        retrieved_chunks = []

        for rank, (doc, score) in enumerate(results, start=1):

            distance = float(score)

            print(f"\nRank: {rank}")
            print(f"FAISS Distance: {distance:.4f}")
            print(f"Content: {doc.page_content[:250]}")

            retrieved_chunks.append({
                "rank": rank,
                "distance": distance,
                "content": doc.page_content
            })

        print(f"\nExpected Keywords: {keywords}")
        print(f"Matched Keywords: {matched_keywords}")
        print(f"Status: {status}")

        evaluation_results.append({
            "id": test["id"],
            "category": test["category"],
            "question": question,
            "expected_answer": test["expected_answer"],
            "matched_keywords": matched_keywords,
            "status": status,
            "retrieved_chunks": retrieved_chunks
        })

    # Calculate retrieval hit rate
    hit_rate = (
        successful_retrievals / total_positive * 100
        if total_positive > 0 else 0
    )

    summary = {
        "total_questions": len(questions),
        "positive_questions": total_positive,
        "successful_retrievals": successful_retrievals,
        "retrieval_hit_rate": round(hit_rate, 2),
        "top_k": TOP_K,
        "results": evaluation_results
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total Questions: {len(questions)}")
    print(f"Positive Questions: {total_positive}")
    print(f"Successful Retrievals: {successful_retrievals}")
    print(f"Retrieval Hit Rate: {hit_rate:.2f}%")
    print(f"\nDetailed results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    evaluate_retrieval()
