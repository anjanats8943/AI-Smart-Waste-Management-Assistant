from agent import waste_agent


print("=" * 55)
print("   AI-BASED SMART WASTE MANAGEMENT ASSISTANT")
print("   RAG + AGENTIC AI")
print("=" * 55)

print("\nType your waste-related question.")
print("Type 'exit' to stop.\n")


while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("\nThank you for using Smart Waste Management Assistant!")
        break

    if not query.strip():
        print("Please enter a question.")
        continue

    result = waste_agent(query)

    print("\nAI Assistant:")
    print("Waste Category:", result["category"])
    print("Guidance:", result["answer"])

    print("\nKnowledge Sources:")
    for source in result["sources"]:
        print("-", source["title"])

    print("\n" + "-" * 55 + "\n")