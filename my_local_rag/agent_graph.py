import re
import sqlite3
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app import (
    call_ollama_chat,
    search_documents,
    generate_polished_answer,
)

from tools import calculator

from mcp_tools import (
    mcp_add_numbers,
    mcp_get_document_list,
)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):
    question: str
    messages: list
    plan: list
    current_step: int
    iteration: int

    tools_used: list
    tool_results: list
    evidence: list

    analysis_result: str
    final_answer: str
    status: str


# ============================================================
# GREETING DETECTION
# ============================================================

def is_greeting(question: str) -> bool:

    q = question.lower().strip()

    greetings = [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    return q in greetings


# ============================================================
# MEMORY QUESTION DETECTION
# ============================================================

def is_memory_question(question: str) -> bool:

    q = question.lower().strip()

    memory_phrases = [
        "what do you remember about me",
        "what do you know about me",
        "do you remember me",
        "do you remember",
        "what did i tell you",
        "what have i told you",
        "remember my",
        "my previous question",
        "my previous questions",
        "our previous conversation",
        "our previous conversations",
        "what did we discuss",
        "what did we talk about",
    ]

    return any(
        phrase in q
        for phrase in memory_phrases
    )


# ============================================================
# DOCUMENT LIST DETECTION
# ============================================================

def is_document_list_question(question: str) -> bool:

    q = question.lower().strip()

    phrases = [
        "what documents",
        "which documents",
        "list documents",
        "show documents",
        "documents do i have",
        "documents i have",
        "documents given",
        "documents i gave",
        "documents uploaded",
        "what files",
        "which files",
        "list files",
        "show files",
        "what pdf",
        "which pdf",
        "list pdf",
        "show pdf",
        "what pdfs",
        "which pdfs",
        "list pdfs",
        "show pdfs",
    ]

    if any(
        phrase in q
        for phrase in phrases
    ):
        return True

    document_words = [
        "document",
        "documents",
        "pdf",
        "pdfs",
        "file",
        "files",
    ]

    list_words = [
        "what",
        "which",
        "list",
        "show",
        "given",
        "uploaded",
    ]

    content_words = [
        "skill",
        "skills",
        "experience",
        "education",
        "qualification",
        "project",
        "projects",
        "analyse",
        "analyze",
        "analysis",
        "mentioned",
        "mention",
    ]

    if (
        any(
            word in q
            for word in document_words
        )
        and
        any(
            word in q
            for word in list_words
        )
        and
        not any(
            word in q
            for word in content_words
        )
    ):
        return True

    return False


# ============================================================
# CALCULATOR DETECTION
# ============================================================

def is_calculation(question: str) -> bool:

    q = question.lower().strip()

    if re.fullmatch(
        r"[\d\s\+\-\*\/\(\)\.\%]+",
        q
    ):
        return True

    if "calculate" in q:
        return True

    if re.search(
        r"\d+\s*(plus|minus|times|divided by)\s*\d+",
        q
    ):
        return True

    return False


# ============================================================
# PLANNER
# ============================================================

def planner_node(state: AgentState):

    question = state["question"]

    q = question.lower()

    plan = []

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if is_greeting(question):

        plan = [
            "greeting"
        ]

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    elif is_memory_question(question):

        plan = [
            "memory"
        ]

    # --------------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------------

    elif is_document_list_question(question):

        plan = [
            "mcp_document_list"
        ]

    # --------------------------------------------------------
    # MULTI-STEP SKILL ANALYSIS
    # --------------------------------------------------------

    elif (
        "skill" in q
        and
        (
            "analyse" in q
            or "analyze" in q
            or "analysis" in q
        )
        and
        (
            "document" in q
            or "documents" in q
            or "pdf" in q
            or "pdfs" in q
            or "file" in q
            or "files" in q
            or "resume" in q
            or "cv" in q
        )
    ):

        plan = [
            "rag",
            "analysis"
        ]

    # --------------------------------------------------------
    # GENERAL DOCUMENT ANALYSIS
    # --------------------------------------------------------

    elif (
        (
            "resume" in q
            or "cv" in q
            or "document" in q
            or "documents" in q
            or "pdf" in q
            or "pdfs" in q
            or "file" in q
            or "files" in q
        )
        and
        (
            "analyse" in q
            or "analyze" in q
            or "analysis" in q
            or "count" in q
            or "how many" in q
        )
    ):

        plan = [
            "rag",
            "analysis"
        ]

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    elif (
        "resume" in q
        or "cv" in q
        or "document" in q
        or "documents" in q
        or "pdf" in q
        or "pdfs" in q
        or "file" in q
        or "files" in q
        or "according to my" in q
        or "in my file" in q
        or "skill" in q
        or "skills" in q
        or "experience" in q
        or "education" in q
        or "qualification" in q
        or "project" in q
        or "projects" in q
    ):

        plan = [
            "rag"
        ]

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------

    elif is_calculation(question):

        plan = [
            "calculator"
        ]

    # --------------------------------------------------------
    # MCP ADDITION
    # --------------------------------------------------------

    elif (
        "add" in q
        and
        len(
            re.findall(
                r"\d+(?:\.\d+)?",
                q
            )
        ) >= 2
    ):

        plan = [
            "mcp_add"
        ]

    # --------------------------------------------------------
    # GENERAL LLM
    # --------------------------------------------------------

    else:

        plan = [
            "llm"
        ]

    print(
        f"\n[Planner selected plan: {plan}]"
    )

    return {
        "plan": plan,
        "current_step": 0,
        "iteration": 0,
        "tools_used": [],
        "tool_results": [],
        "evidence": [],
        "analysis_result": "",
        "final_answer": "",
        "status": "planning",
    }


# ============================================================
# NEW QUESTION
# ============================================================

def new_question_node(state: AgentState):

    question = state["question"]

    messages = state.get(
        "messages",
        []
    )

    messages = messages + [
        {
            "role": "user",
            "content": question,
        }
    ]

    return {
        "messages": messages,
        "iteration": 0,
        "current_step": 0,
        "tools_used": [],
        "tool_results": [],
        "evidence": [],
        "analysis_result": "",
        "final_answer": "",
        "status": "running",
    }


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(state: AgentState):

    question = state["question"]

    plan = state.get(
        "plan",
        []
    )

    current_step = state.get(
        "current_step",
        0
    )

    messages = state.get(
        "messages",
        []
    )

    iteration = state.get(
        "iteration",
        0
    ) + 1

    print(
        f"Agent iteration: {iteration}"
    )

    if current_step >= len(plan):

        return {
            "iteration": iteration,
            "status": "finished",
        }

    action = plan[current_step]

    print(
        f"[Executing plan step "
        f"{current_step + 1}: {action}]"
    )

    # ========================================================
    # GREETING
    # ========================================================

    if action == "greeting":

        response = call_ollama_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Respond briefly and naturally "
                        "to the user's greeting."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
        )

        return {
            "messages": messages + [
                {
                    "role": "assistant",
                    "content": response,
                }
            ],
            "current_step": len(plan),
            "iteration": iteration,
            "status": "finished",
        }

    # ========================================================
    # MEMORY
    # ========================================================

    if action == "memory":

        response = call_ollama_chat(
            messages=messages
        )

        return {
            "messages": messages + [
                {
                    "role": "assistant",
                    "content": response,
                }
            ],
            "current_step": len(plan),
            "iteration": iteration,
            "status": "finished",
        }

    # ========================================================
    # RAG
    # ========================================================

    if action == "rag":

        print(
            f"[Searching documents: {question}]"
        )

        # IMPORTANT:
        # Your existing app.py search_documents()
        # does not accept top_k.
        results = search_documents(
            question
        )

        evidence = []

        for result in results:

            if isinstance(
                result,
                dict
            ):

                evidence.append(
                    result
                )

            else:

                evidence.append(
                    {
                        "text": str(result)
                    }
                )

        return {
            "evidence": evidence,

            "tool_results": (
                state.get(
                    "tool_results",
                    []
                )
                +
                [
                    {
                        "tool": "rag",
                        "result": evidence,
                    }
                ]
            ),

            "tools_used": (
                state.get(
                    "tools_used",
                    []
                )
                +
                ["rag"]
            ),

            "current_step": (
                current_step + 1
            ),

            "iteration": iteration,

            "status": "running",
        }

    # ========================================================
    # CALCULATOR
    # ========================================================

    if action == "calculator":

        print(
            f"[Calculator: {question}]"
        )

        result = calculator(
            question
        )

        print(
            f"[Calculator result: {result}]"
        )

        return {
            "tool_results": (
                state.get(
                    "tool_results",
                    []
                )
                +
                [
                    {
                        "tool": "calculator",
                        "result": result,
                    }
                ]
            ),

            "tools_used": (
                state.get(
                    "tools_used",
                    []
                )
                +
                ["calculator"]
            ),

            "current_step": (
                current_step + 1
            ),

            "iteration": iteration,

            "status": "running",
        }

    # ========================================================
    # MCP ADD
    # ========================================================

    if action == "mcp_add":

        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            question
        )

        if len(numbers) >= 2:

            a = float(
                numbers[0]
            )

            b = float(
                numbers[1]
            )

            result = mcp_add_numbers(
                a,
                b
            )

            print(
                f"[MCP add_numbers result: {result}]"
            )

            return {
                "tool_results": (
                    state.get(
                        "tool_results",
                        []
                    )
                    +
                    [
                        {
                            "tool":
                                "mcp_add_numbers",
                            "result":
                                result,
                        }
                    ]
                ),

                "tools_used": (
                    state.get(
                        "tools_used",
                        []
                    )
                    +
                    ["mcp_add_numbers"]
                ),

                "current_step": (
                    current_step + 1
                ),

                "iteration": iteration,

                "status": "running",
            }

    # ========================================================
    # MCP DOCUMENT LIST
    # ========================================================

    if action == "mcp_document_list":

        print(
            "[Calling MCP: get_document_list]"
        )

        result = mcp_get_document_list()

        print(
            f"[MCP document list result: {result}]"
        )

        return {
            "tool_results": (
                state.get(
                    "tool_results",
                    []
                )
                +
                [
                    {
                        "tool":
                            "mcp_get_document_list",
                        "result":
                            result,
                    }
                ]
            ),

            "tools_used": (
                state.get(
                    "tools_used",
                    []
                )
                +
                ["mcp_get_document_list"]
            ),

            "current_step": (
                current_step + 1
            ),

            "iteration": iteration,

            "status": "running",
        }

    # ========================================================
    # ANALYSIS
    # ========================================================

    if action == "analysis":

        evidence = state.get(
            "evidence",
            []
        )

        evidence_text = "\n\n".join(
            str(item)
            for item in evidence
        )

        print(
            "[Analysing retrieved documents]"
        )

        analysis_prompt = f"""
You are analysing information retrieved
from the user's documents.

User question:

{question}

Retrieved document information:

{evidence_text}

Analyse the information and answer
the user's request.

If skills are requested:

- identify the skills
- explain them briefly
- mention the document or page
  when available

Do not invent information that is
not present in the retrieved documents.
"""

        analysis_result = call_ollama_chat(
            messages=[
                {
                    "role": "user",
                    "content":
                        analysis_prompt,
                }
            ]
        )

        return {
            "analysis_result":
                analysis_result,

            "tool_results": (
                state.get(
                    "tool_results",
                    []
                )
                +
                [
                    {
                        "tool": "analysis",
                        "result":
                            analysis_result,
                    }
                ]
            ),

            "tools_used": (
                state.get(
                    "tools_used",
                    []
                )
                +
                ["analysis"]
            ),

            "current_step": (
                current_step + 1
            ),

            "iteration": iteration,

            "status": "running",
        }

    # ========================================================
    # GENERAL LLM
    # ========================================================

    if action == "llm":

        response = call_ollama_chat(
            messages=messages
        )

        return {
            "messages": messages + [
                {
                    "role": "assistant",
                    "content": response,
                }
            ],

            "current_step": len(plan),

            "iteration": iteration,

            "status": "finished",
        }

    return {
        "current_step": len(plan),
        "iteration": iteration,
        "status": "finished",
    }


# ============================================================
# ROUTER
# ============================================================

def should_continue(state: AgentState):

    plan = state.get(
        "plan",
        []
    )

    current_step = state.get(
        "current_step",
        0
    )

    if current_step >= len(plan):

        return "final"

    return "agent"


# ============================================================
# FINAL ANSWER
# ============================================================

def final_answer_node(state: AgentState):

    question = state["question"]

    plan = state.get(
        "plan",
        []
    )

    evidence = state.get(
        "evidence",
        []
    )

    tool_results = state.get(
        "tool_results",
        []
    )

    analysis_result = state.get(
        "analysis_result",
        ""
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    if analysis_result:

        answer = analysis_result

    # ========================================================
    # CALCULATOR
    # ========================================================

    elif "calculator" in plan:

        result = None

        for item in tool_results:

            if item.get(
                "tool"
            ) == "calculator":

                result = item.get(
                    "result"
                )

        answer = (
            f"The result of {question} "
            f"is {result}."
        )

    # ========================================================
    # MCP ADD
    # ========================================================

    elif "mcp_add" in plan:

        result = None

        for item in tool_results:

            if item.get(
                "tool"
            ) == "mcp_add_numbers":

                result = item.get(
                    "result"
                )

        answer = (
            f"The answer is {result}."
        )

    # ========================================================
    # MCP DOCUMENT LIST
    # ========================================================

    elif "mcp_document_list" in plan:

        result = None

        for item in tool_results:

            if item.get(
                "tool"
            ) == "mcp_get_document_list":

                result = item.get(
                    "result"
                )

        answer = (
            "You have these documents:\n\n"
            f"{result}"
        )

    # ========================================================
    # RAG
    # ========================================================

    elif evidence:

        answer = generate_polished_answer(
            question,
            evidence,
            state.get(
                "messages",
                []
            )
        )

    # ========================================================
    # MESSAGE
    # ========================================================

    else:

        messages = state.get(
            "messages",
            []
        )

        answer = ""

        for message in reversed(
            messages
        ):

            if (
                isinstance(
                    message,
                    dict
                )
                and
                message.get(
                    "role"
                ) == "assistant"
            ):

                answer = message.get(
                    "content",
                    ""
                )

                break

        if not answer:

            answer = (
                "I couldn't generate an answer."
            )

    print(
        "\nAssistant:"
    )

    print(
        answer
    )

    return {
        "final_answer": answer,
        "status": "finished",
    }


# ============================================================
# BUILD GRAPH
# ============================================================

graph_builder = StateGraph(
    AgentState
)

graph_builder.add_node(
    "new_question",
    new_question_node
)

graph_builder.add_node(
    "planner",
    planner_node
)

graph_builder.add_node(
    "agent",
    agent_node
)

graph_builder.add_node(
    "final",
    final_answer_node
)


# ============================================================
# GRAPH FLOW
# ============================================================

graph_builder.add_edge(
    START,
    "new_question"
)

graph_builder.add_edge(
    "new_question",
    "planner"
)

graph_builder.add_edge(
    "planner",
    "agent"
)

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "agent": "agent",
        "final": "final",
    }
)

graph_builder.add_edge(
    "final",
    END
)


# ============================================================
# SQLITE MEMORY
# ============================================================

sqlite_connection = sqlite3.connect(
    "langgraph_memory.db",
    check_same_thread=False
)

memory = SqliteSaver(
    sqlite_connection
)

graph = graph_builder.compile(
    checkpointer=memory
)


# ============================================================
# RUN GRAPH
# ============================================================

def run_graph_agent(question):

    initial_state = {
        "question": question,
        "messages": [],
    }

    config = {
        "configurable": {
            "thread_id": "user-1"
        }
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Local Agentic RAG started."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "Goodbye."
            )

            break

        if not question:

            continue

        try:

            result = run_graph_agent(
                question
            )

            print(
                f"\nPlan: "
                f"{result.get('plan', [])}"
            )

            print(
                f"Tools used: "
                f"{result.get('tools_used', [])}"
            )

            print(
                f"Status: "
                f"{result.get('status', 'unknown')}"
            )

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)