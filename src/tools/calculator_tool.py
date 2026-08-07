import numexpr
def calculator_tool(expression: str) -> str:
    """Safely evaluates a math expression. Deliberately NOT using Python's
    eval() — numexpr only understands arithmetic syntax, it has no concept
    of import, exec, or arbitrary Python, so there's no code-execution risk
    even without a sandbox."""
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


if __name__ == "__main__":
    print(calculator_tool("175e9 * 2 / (1024**3)"))  # GB test from q05
    print(calculator_tool("15/100 * 2340"))            # q13
    print(calculator_tool("import os"))                # should safely error, not execute