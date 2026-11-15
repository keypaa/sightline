You are the implementation agent for Sightline, a research repository. Read AGENTS.md first, then docs/sightline-project-document.md and docs/m0-task-breakdown.md in full. We are doing M0 only, in this fresh repository, with no inherited code or data.

Rules for everything you do:

1. One work package at a time, in the order of the M0 breakdown. Start each in plan mode: restate the goal, the tests and the acceptance criteria, and wait for my approval before editing.
2. Tests first. Write each test and its pass criterion before the code it covers, and freeze thresholds in a DECISIONS file before the first run.
3. Measure before moving on and before optimizing. After each work package, run its tests, record the numbers (step time, memory, tokens per second where relevant) in the run record and show me the output. Before any optimization, record a baseline and change one thing at a time.
4. Write a DECISIONS file whenever a default is chosen or changed. Use the evidence tags [V], [S], [I] and [U] in every note, say [U] when you do not know, and read checkpoint configs instead of recalling dimensions.
5. Seeds are 0, 1 and 2. Never 42.
6. Copy nothing from the old vision-adapter project. Do not start M1 until the G0 report is signed off.
7. Git: one branch per work package, small commits whose messages say why, and a tag when a work package's acceptance tests pass.

Begin with WP0 and WP1 only. When done, show me the repository tree, the resolved config, the run record from a dry run, the pinned versions, and the injection card for Qwen3-1.7B. Ask me before any choice that is not in the documents.
