# AgentScope Sample Agents

Welcome to the **AgentScope Sample Agents** repository! 🎯
This repository provides **ready-to-use Python sample agents** built on top of:

- [AgentScope](https://github.com/agentscope-ai/agentscope)
- [AgentScope Runtime](https://github.com/agentscope-ai/agentscope-runtime)

The examples cover a wide range of use cases — from lightweight command-line agents to **full-stack deployable applications** with both backend and frontend.

------

## 📖 About AgentScope & AgentScope Runtime

### **AgentScope**

AgentScope is a multi-agent framework designed to provide a **simple and efficient** way to build **LLM-powered agent applications**. It offers abstractions for defining agents, integrating tools, managing conversations, and orchestrating multi-agent workflows.

### **AgentScope Runtime**

AgentScope Runtime is a **comprehensive runtime framework** that addresses two key challenges in deploying and operating agents:

1. **Effective Agent Deployment** – Scalable deployment and management of agents across environments.
2. **Sandboxed Tool Execution** – Secure, isolated execution of tools and external actions.

It includes **context management** and **secure sandboxing**, and can be used with **AgentScope** or other agent frameworks.

------

## ✨ Getting Started

- All samples are **Python-based**.
- Samples are organized **by functional use case**.
- Some samples use only **AgentScope** (pure Python agents).
- Others use **both AgentScope and AgentScope Runtime** to implement **full-stack deployable applications** with frontend + backend.
- Full-stack runtime versions have folder names ending with:
  **`_fullstack_runtime`**

> 📌 **Before running** any example, check its `README.md` for installation and execution instructions.

### Install Requirements

- [AgentScope Documentation](https://doc.agentscope.io/)
- [AgentScope Runtime Documentation](https://runtime.agentscope.io/)

------

## 🌳 Repository Structure

```bash
├── browser_use/
│   ├── agent_browser/                      # Pure Python browser agent
│   └── browser_use_fullstack_runtime/      # Full-stack runtime version with frontend/backend
│
├── deep_research/
│   ├── agent_deep_research/                # Pure Python multi-agent research
│   └── qwen_langgraph_search_fullstack_runtime/    # Full-stack runtime-enabled research app
│
├── games/
│   └── game_werewolves/                    # Role-based social deduction game
│
├── conversational_agents/
│   ├── chatbot/                            # Chatbot application
│   ├── chatbot_fullstack_runtime/          # Runtime-powered chatbot with UI
│   ├── multiagent_conversation/            # Multi-agent dialogue scenario
│   └── multiagent_debate/                  # Agents engaging in debates
│
├── evaluation/
│   └── ace_bench/                          # Benchmarks and evaluation tools
│
├── functionality/
│   ├── long_term_memory_mem0/              # Long-term memory integration
│   ├── mcp/                                # Memory/Context Protocol demo
│   ├── plan/                               # Plan with ReAct Agent
│   ├── rag/                                # RAG in AgentScope
│   ├── session_with_sqlite/                # Persistent conversation with SQLite
│   ├── stream_printing_messages/           # Streaming and printing messages
│   ├── structured_output/                  # Structured output parsing and validation
│   ├── multiagent_concurrent/              # Concurrent multi-agent task execution
│   └── meta_planner_agent/                  # Planning agent with tool orchestration
│
├── agentic_rl/
│   └── react_agent/                        # Training ReAct agent with RL
│
└── README.md
```

------

## 📌 Example List

| Category                | Example Folder                                        | Uses AgentScope | Uses Runtime | Description                                      |
| ----------------------- |-------------------------------------------------------| --------------- | ------------ |--------------------------------------------------|
| **Browser Use**         | browser_use/agent_browser                             | ✅               | ❌            | Command-line browser automation using AgentScope |
|                         | browser_use/browser_use_fullstack_runtime             | ✅               | ✅            | Full-stack browser automation with UI & sandbox  |
| **Deep Research**       | deep_research/agent_deep_research                     | ✅               | ❌            | Multi-agent research pipeline                    |
|                         | deep_research/qwen_langgraph_search_fullstack_runtime | ❌               | ✅            | Full-stack deep research app                     |
| **Games**               | games/game_werewolves                                 | ✅               | ❌            | Multi-agent roleplay game                        |
| **Conversational Apps** | conversational_agents/chatbot_fullstack_runtime       | ✅               | ✅            | Chatbot application with frontend/backend        |
|                         | conversational_agents/chatbot                         | ✅               | ❌            |                                                  |
|                         | conversational_agents/multiagent_conversation         | ✅               | ❌            | Multi-agent dialogue scenario                    |
|                         | conversational_agents/multiagent_debate               | ✅               | ❌            | Agents engaging in debates                       |
| **Evaluation**          | evaluation/ace_bench                                  | ✅               | ❌            | Benchmarks with ACE Bench                        |
| **Functionality Demos** | functionality/long_term_memory_mem0                   | ✅               | ❌            | Long-term memory with mem0 support               |
|                         | functionality/mcp                                     | ✅               | ❌            | Memory/Context Protocol demo                     |
|                         | functionality/session_with_sqlite                     | ✅               | ❌            | Persistent context with SQLite                   |
|                         | functionality/structured_output                       | ✅               | ❌            | Structured data extraction and validation        |
|                         | functionality/multiagent_concurrent                   | ✅               | ❌            | Concurrent task execution by multiple agents     |
|                         | functionality/meta_planner_agent                      | ✅               | ❌            | Planning agent with tool orchestration           |
|                         | functionality/plan                                    | ✅               | ❌            | Task planning with ReAct agent                   |
|                         | functionality/rag                                     | ✅               | ❌            | Retrieval-Augmented Generation (RAG) integration |
|                         | functionality/stream_printing_messages                | ✅               | ❌            | Real-time message streaming and printing         |
| **Agentic RL**          | agentic_rl/react_agent                                | ✅               | ❌            | Training ReAct agent with Reinforcement Learning  |

------

## ℹ️ Getting Help

If you:

- Need installation help
- Encounter issues
- Want to understand how a sample works

Please:

1. Read the sample-specific `README.md`.
2. File a [GitHub Issue](https://github.com/agentscope-ai/agentscope-samples/issues).
3. Join the community discussions.

------

## 🤝 Contributing

We welcome contributions such as:

- Bug reports
- New feature requests
- Documentation improvements
- Code contributions

See the [Contributing Guidelines](https://github.com/agentscope-ai/agentscope-samples/CONTRIBUTING.md) for details.

------

## 📄 License

This project is licensed under the **Apache 2.0 License** – see the [LICENSE](https://github.com/agentscope-ai/agentscope-samples/LICENSE) file for details.

------

## ⚠️ Disclaimer

- This is not an officially supported product.
- For **demonstration purposes only** — not intended for production use.

------

## 🔗 Resources

- [AgentScope Documentation](https://doc.agentscope.io/)
- [AgentScope Runtime Documentation](https://runtime.agentscope.io/)
- [AgentScope GitHub Repository](https://github.com/agentscope-ai/agentscope)
- [AgentScope Runtime GitHub Repository](https://github.com/agentscope-ai/agentscope-runtime)