---
title: Case Study 1
course: intelligent-agents
---
## Case Study: AI Safety Constraints

In the design of my academic research tool, I chose to keep my files and LLMs local for the sake of safety (Clark, 2026). I added additional buttons for stopping, pausing, and restarting the AI agents not only for convenience but for safety reasons as well. If the AI agent is outputting inappropriate or unsafe content shown in their logs or JSON file, the user would be able to see that and stop the AI agent from continuing to work (Hughes, 2026). I thought of these buttons working like “kill switches” and giving the user more control over the application. I gave the AI agents a lot of autonomy in terms of their decision making. I have not tested if they could go beyond the bounds of their programming, but it could be possible (Hughes, 2026). When I was initially designing the tool, the AI agents were not doing exactly what I wanted at the start.

![Academic Research Tool interface with the send button and the Pause, Stop and Reset buttons highlighted](/docs/IntelligentAgents/casestudy1-interface.png)

I did not want to make my application reliant on connections and APIs as a safety measure to keep everything local to one’s computer (Clark, 2026). The design of having the logs and markdown files stored to the user’s computer is also keeping the AI agents contained to one environment. For myself uploading my assignment for submission, I had to remove my personal computer details and APIs for my own safety (Hughes, 2026). I also would not upload that information to my GitHub e-portfolio and make myself vulnerable to attacks, or risk my API being abused (Hughes, 2026).

![RunControl class in control.py with the stop, pause and resume methods](/docs/IntelligentAgents/casestudy1-control.png)

### References

Clark, B. (2026) *AI agent security best practices guide*. Available at: <https://www.ibm.com/think/tutorials/ai-agent-security> (Accessed: 18 October 2026).

Hughes, J. (2026) *How we built Claude Code auto mode: a safer way to skip permissions*. Available at: <https://www.anthropic.com/engineering/claude-code-auto-mode> (Accessed: 18 October 2026).
