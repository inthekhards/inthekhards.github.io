---
title: Case Study 2
course: intelligent-agents
date: 2026-10-18 12:00:00 +0000
---
## Case Study: Bias in Autonomous Decision-Making

When I was in the development stage of the academic research tool project, I noticed a reoccurring problem of the tool presenting foreign language articles in the results. I could not figure out why the first author’s name was appearing in Greek or Cyrillic script. I figured it must have been a bug happening in multiple AI agent processes.

![Results file with a reference whose first author's name is in Cyrillic script](/docs/IntelligentAgents/casestudy2-results-greek.png)

![Results file with a reference whose first author's name is in Greek script](/docs/IntelligentAgents/casestudy2-results-cyrillic.png)

I found it weird that the articles appeared to be in English, but the author names were not. I decided to add language filters and selection after seeing this happen in the results. I gave the AI agents a bias towards choosing English language academic articles, or at least a bias towards the language it is being used in (Holdsworth, 2023). The filter to correct this is called “language_problem”, found in app/languages.py.

![The language_problem function in languages.py](/docs/IntelligentAgents/casestudy2-language-problem.png)

Then, the processing agent (proc.py) calls for this filter when processing the raw research results that were retrieved from APIs from the retrieval agent (ret.py).

![The in_language method in proc.py calling language_problem](/docs/IntelligentAgents/casestudy2-in-language.png)

This was not an issue I was expecting to happen when designing this program with my team (Holdsworth, 2023). I am glad that I caught it while testing how well the AI agents worked. It is another aspect of AI programming for me to think about in my design for my future AI research project module.

The code can be found here: [Unit 11: Academic Research Tool](/2026/10/13/IAUnit11.html)

### Reference

Holdsworth, J. (2023) *What is AI bias?* IBM. Available at: <https://www.ibm.com/think/topics/ai-bias> (Accessed: 18 October 2026).
