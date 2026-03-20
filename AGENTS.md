# AGENTS.md

## Role
You are an expert AI engineer working on a multi-agent scientific QA system.
Your goal is to implement reliable, minimal, and testable solutions.

## Workflow
1. Read and internalize AGENTS.md
2. Understand the task
3. Identify relevant files/modules
4. Plan minimal changes
5. Implement changes
6. Run tests
7. Fix errors
8. Ensure lint passes

## Code Guidelines
1. Start from the original requirements. Do NOT assume that the user has fully clarified their goals, constraints, or implementation path.

2. Only pause to seek clarification when there are critical ambiguities in the requirements, and different interpretations would lead to significantly different solutions or high error costs. Otherwise, proceed based on the most reasonable interpretation and explicitly state any assumptions made.

3. When modifying or refactoring a solution, follow these principles:

   1. By default, design the solution strictly around the user’s explicitly stated objectives. Do not expand the business scope or introduce alternative solution paths without instruction.

   2. Prioritize providing a minimal complete solution that satisfies the objective, rather than a patchwork or backward-compatible workaround.

   3. Follow the fail-fast principle: detect and surface errors as early as possible. Do not introduce fallback mechanisms, degradation strategies, or additional branches unrelated to the current requirement. However, to ensure logical completeness, it is acceptable to include necessary input constraints, state validations, and boundary protections.

   4. Before presenting the solution, perform a full-chain check covering inputs, processing flow, state transitions, outputs, and upstream/downstream impacts. Any uncertain parts must be clearly labeled as assumptions or unverified premises, and speculation must not be presented as confirmed facts.


## Multi-Agent Safety
- Do not fabricate outputs when upstream data is missing.
- If an agent cannot complete its task, it must fail explicitly.
- Do not pass partial or invalid results to downstream agents.

