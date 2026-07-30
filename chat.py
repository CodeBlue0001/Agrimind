"""
AgroCulture AI — Interactive Chat Interface
=============================================
Color-coded terminal chat for interacting with the AgroCulture AI system.

Usage:
    python chat.py          # Start interactive chat (full system)
    python chat.py --test   # Run automated test queries
    python chat.py --no-kb  # Skip knowledge base (faster startup)
    python chat.py --no-llm # Skip traditional LLM backend

Commands:
    /help    — Show usage instructions
    /models  — Show loaded model status
    /llm     — Show traditional LLM backend status
    /reset   — Clear conversation history
    /quit    — Exit the chat
"""

import os
import sys
import argparse
import time

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "model", "llm"))

try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback: no-op color codes
    class _NoColor:
        def __getattr__(self, name):
            return ""
    Fore = _NoColor()
    Style = _NoColor()
    Back = _NoColor()


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

BANNER = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     🌾  AgroCulture AI  🌾                                       ║
    ║     ━━━━━━━━━━━━━━━━━━━━                                         ║
    ║     Intelligent Farming Assistant                                ║
    ║                                                                  ║
    ║     Crop Prediction • Fertilizer Advice • Soil Analysis          ║
    ║     Disease Remediation • Yield Forecasting                      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
"""

EXAMPLE_PROMPTS = [
    "What crop should I grow with N=80, P=40, K=40, temp 20°C, humidity 50%, pH 6.5, rainfall 75mm in loamy soil?",
    "What fertilizer for sugarcane? My soil has N=40, P=10, K=80",
    "Analyze my soil: N=138, P=8.6, K=560, pH=7.46",
    "How to treat white rot in onion?",
    "Predict wheat production in Punjab for 2026",
]


def print_banner():
    """Print the welcome banner."""
    print(f"{Fore.GREEN}{BANNER}{Style.RESET_ALL}")


def print_status(text: str, color=None):
    """Print a status message."""
    c = color or Fore.CYAN
    print(f"{c}{text}{Style.RESET_ALL}")


def print_user(text: str):
    """Print user message."""
    print(f"\n{Fore.WHITE}{Style.BRIGHT}👤 You: {text}{Style.RESET_ALL}")


def print_ai(text: str):
    """Print AI response with formatting."""
    print(f"\n{Fore.GREEN}🤖 AgroCulture AI:{Style.RESET_ALL}")
    # Color different parts of the response
    for line in text.split("\n"):
        if line.strip().startswith("🎯") or line.strip().startswith("**"):
            print(f"  {Fore.YELLOW}{line}{Style.RESET_ALL}")
        elif line.strip().startswith("⚠️") or line.strip().startswith("🔴"):
            print(f"  {Fore.RED}{line}{Style.RESET_ALL}")
        elif line.strip().startswith("✅") or line.strip().startswith("🟢"):
            print(f"  {Fore.GREEN}{line}{Style.RESET_ALL}")
        elif line.strip().startswith("📊") or line.strip().startswith("📈") or line.strip().startswith("📚"):
            print(f"  {Fore.CYAN}{line}{Style.RESET_ALL}")
        elif line.strip().startswith("💊") or line.strip().startswith("💰"):
            print(f"  {Fore.MAGENTA}{line}{Style.RESET_ALL}")
        elif line.strip().startswith("•") or line.strip().startswith("-"):
            print(f"  {Fore.WHITE}{line}{Style.RESET_ALL}")
        else:
            print(f"  {line}")
    print()


def print_separator():
    """Print a visual separator."""
    print(f"{Fore.BLUE}{'─' * 70}{Style.RESET_ALL}")


def print_examples():
    """Print example prompts."""
    print(f"\n{Fore.CYAN}💡 Try these example queries:{Style.RESET_ALL}")
    for i, prompt in enumerate(EXAMPLE_PROMPTS, 1):
        print(f"  {Fore.WHITE}{i}. {prompt}{Style.RESET_ALL}")
    print()


# ---------------------------------------------------------------------------
# Main Chat Loop
# ---------------------------------------------------------------------------

def run_interactive(load_kb: bool = True, load_trad_llm: bool = True):
    """Run the interactive chat loop."""
    print_banner()
    print_status("  Initializing AgroCulture AI system...\n")

    # Import and initialize the LLM
    from agro_llm import AgroCultureLLM
    llm = AgroCultureLLM(load_kb=load_kb, load_trad_llm=load_trad_llm)

    print()
    print_separator()
    print(llm.model_status)
    print_separator()
    print_examples()
    print_status("  Type your question below. Use /help for commands, /quit to exit.\n")

    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.WHITE}{Style.BRIGHT}👤 You: {Style.RESET_ALL}").strip()

            if not user_input:
                continue

            # Handle commands
            cmd = user_input.lower()

            if cmd in ["/quit", "/exit", "/q", "exit", "quit", "bye"]:
                print_status("\n  🌾 Thank you for using AgroCulture AI! Happy farming! 🌾\n", Fore.GREEN)
                break

            elif cmd in ["/help", "/h"]:
                response = llm.chat("/help")
                print_ai(response)
                continue

            elif cmd in ["/models", "/status"]:
                print(f"\n{llm.model_status}\n")
                continue

            elif cmd in ["/llm", "/llm-status"]:
                if llm._trad_llm:
                    print(f"\n{Fore.CYAN}🤖 Traditional LLM Backend Status:{Style.RESET_ALL}")
                    print(llm._trad_llm.status)
                    stats = llm._trad_llm.get_unprocessable_stats()
                    print(f"\n  📋 Unprocessable queries logged: {stats['total']}")
                    for reason, count in stats['by_reason'].items():
                        print(f"     {reason}: {count}")
                    print()
                else:
                    print_status("  Traditional LLM not loaded.\n", Fore.YELLOW)
                continue

            elif cmd in ["/reset", "/clear"]:
                llm.reset()
                print_status("  🔄 Conversation reset. Start fresh!\n")
                continue

            elif cmd in ["/examples", "/ex"]:
                print_examples()
                continue

            # Process the message
            print_user(user_input)

            start_time = time.time()
            response = llm.chat(user_input)
            elapsed = time.time() - start_time

            print_ai(response)
            print_status(f"  ⏱ Response time: {elapsed:.2f}s", Fore.BLUE)
            print_separator()

        except KeyboardInterrupt:
            print_status("\n\n  🌾 Goodbye! Happy farming! 🌾\n", Fore.GREEN)
            break
        except EOFError:
            break
        except Exception as e:
            print(f"\n{Fore.RED}  ⚠️ Error: {str(e)}{Style.RESET_ALL}\n")


# ---------------------------------------------------------------------------
# Test Mode
# ---------------------------------------------------------------------------

def run_test():
    """Run automated test queries."""
    print_banner()
    print_status("  Running automated tests...\n")

    from agro_llm import AgroCultureLLM
    llm = AgroCultureLLM(load_kb=True, load_trad_llm=True)

    print(f"\n{llm.model_status}\n")
    print_separator()

    test_cases = [
        ("Greeting",                   "Hello"),
        ("Crop Prediction",             "What crop should I grow with N=80, P=40, K=40, temperature 20°C, humidity 50%, pH 6.5, rainfall 75mm in loamy soil?"),
        ("Fertilizer (Rule Engine)",    "What fertilizer for sugarcane? My soil has N=40, P=10, K=80"),
        ("Soil Analysis",               "Analyze my soil: N=138, P=8.6, K=560, pH=7.46"),
        ("Disease Query",               "How to treat white rot disease in onion?"),
        ("General QA",                  "Tell me about rice cultivation"),
        ("Missing Fields",              "What crop should I grow?"),
        ("Unprocessable — OOD",         "Who won the IPL 2024?"),
        ("Unprocessable — Too Vague",   "hi"),
        ("Unprocessable — Border Agri", "How do I get a KCC loan?"),
        ("LLM QA — Organic Farming",   "Tell me about organic farming practices"),
        ("LLM QA — IPM",               "What is Integrated Pest Management?"),
    ]

    passed = 0
    failed = 0

    for name, query in test_cases:
        print(f"\n{'=' * 70}")
        print(f"TEST: {name}")
        print(f"QUERY: {query}")
        print(f"{'=' * 70}")

        try:
            start = time.time()
            response = llm.chat(query)
            elapsed = time.time() - start

            print(response)
            print(f"\n⏱ {elapsed:.2f}s")

            if response and len(response) > 10:
                print(f"{Fore.GREEN}✅ PASS{Style.RESET_ALL}")
                passed += 1
            else:
                print(f"{Fore.RED}❌ FAIL (empty/short response){Style.RESET_ALL}")
                failed += 1

        except Exception as e:
            print(f"{Fore.RED}❌ FAIL: {str(e)}{Style.RESET_ALL}")
            failed += 1

        llm.reset()

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AgroCulture AI — Interactive Chat")
    parser.add_argument("--test",   action="store_true", help="Run automated test queries")
    parser.add_argument("--no-kb",  action="store_true", help="Skip loading knowledge base")
    parser.add_argument("--no-llm", action="store_true", help="Skip traditional LLM backend")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_interactive(load_kb=not args.no_kb, load_trad_llm=not args.no_llm)


if __name__ == "__main__":
    main()
