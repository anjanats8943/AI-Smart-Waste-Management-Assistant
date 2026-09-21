from rag_engine import retrieve_rag_information, generate_rag_answer


def waste_agent(user_query):

    # Agent Step 1: Retrieve relevant knowledge
    retrieved = retrieve_rag_information(
        user_query,
        top_k=2
    )

    # Agent Step 2: Identify waste category
    category = retrieved[0]["title"]

    # Agent Step 3: Generate AI answer
    result = generate_rag_answer(user_query)

    # Agent Step 4: Ground the response
    # Use the retrieved knowledge as the main answer
    grounded_answer = retrieved[0]["content"]

    # Agent Step 5: Add safety guidance when needed
    safety_note = ""

    if category in [
        "E-WASTE",
        "HAZARDOUS HOUSEHOLD WASTE"
    ]:

        safety_note = (
            " Follow local waste-management authority "
            "guidelines for safe disposal."
        )

    # Final agent response
    return {
        "category": category,
        "answer": grounded_answer + safety_note,
        "sources": retrieved
    }


# Test the Agent
if __name__ == "__main__":

    query = input(
        "Enter your waste-related question: "
    )

    result = waste_agent(query)

    print("\n--- Smart Waste Management Assistant ---")

    print(
        "Waste Category:",
        result["category"]
    )

    print(
        "Guidance:",
        result["answer"]
    )

    print("\nKnowledge Sources:")

    for source in result["sources"]:

        print(
            "-",
            source["title"]
        )