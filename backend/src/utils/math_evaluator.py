import math
import logging

logger = logging.getLogger(__name__)

# Allowed functions and constants for safe evaluation
ALLOWED_NAMES = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
    "pi": math.pi,
    "e": math.e,
}

def safe_eval_t(expr: str, t: float, default_val: float) -> float:
    """
    Safely evaluates a mathematical expression using 't' as the independent variable.
    
    Args:
        expr: The math expression as a string (e.g., "t**2", "sin(t * pi)").
        t: The independent variable (usually 0.0 to 1.0).
        default_val: The value to return if evaluation fails or expr is empty.
        
    Returns:
        The result of the evaluation as a float.
    """
    if not expr or not isinstance(expr, str) or expr.strip() == "":
        return default_val
        
    # Prepare context
    context = ALLOWED_NAMES.copy()
    context["t"] = t
    
    try:
        # Use eval with restricted globals and locals
        # We explicitly clear __builtins__ for safety
        result = eval(expr, {"__builtins__": {}}, context)
        return float(result)
    except Exception as e:
        logger.error(f"Error evaluating expression '{expr}' with t={t}: {e}")
        return default_val
