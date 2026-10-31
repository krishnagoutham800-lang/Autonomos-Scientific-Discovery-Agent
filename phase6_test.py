from agents.research_workflow import run_research_workflow


question = input(
    "Enter your scientific research question: "
)

report = run_research_workflow(question)

print("\n" + "=" * 60)
print(report)
print("=" * 60)