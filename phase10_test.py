from agents.evidence_validator import validate_research


claims = [
    "Deep learning can help identify patterns in medical images.",

    "Artificial intelligence always produces more accurate diagnoses than human doctors.",

    "Machine learning can support early disease prediction."
]


report = validate_research(claims)


print("\n========== EVIDENCE VALIDATION REPORT ==========\n")

for result in report["results"]:

    print("Claim:", result["claim"])
    print("Status:", result["status"])
    print("Reason:", result["reason"])

    for source in result["sources"]:

        print("Paper:", source["title"])
        print("Year:", source["year"])
        print("Page:", source["page"])
        print("Quote verified:", source["quote_verified"])
        print("Evidence:", source["evidence_quote"])

    print("-" * 60)