import sys
import os
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

# add current dir to the python path so we can import common
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/")

with open("utils/arc_utils/common.py", "r") as f:
    COMMON_LIBRARY_CODE = f.read()


def wrapped_execute_transformation(args):
    return execute_transformation(*args)


def execute_transformation(source, input_grid, timeout=1, function_name="main", verbose=True):

    input_grid = np.array(input_grid)

    global_vars = {}

    def execute_code(source_code, return_var_name, global_vars):

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in ["os", "sys"]:
                raise ImportError(f"Import of '{name}' is not allowed")
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins = {k: v for k, v in __builtins__.items() if k not in ["exit", "quit"]}
        safe_builtins["__import__"] = safe_import

        safe_globals = {
            "__builtins__": safe_builtins,
        }
        exec(source_code, safe_globals)
        if return_var_name not in safe_globals and verbose:
            print(f"Error: {return_var_name} not found in global_vars")
            return None
        ret = safe_globals[return_var_name]
        return ret

    n, m = input_grid.shape
    make_input = f"input_grid = np.zeros(({n}, {m}), dtype=int)\n"
    for i in range(n):
        for j in range(m):
            make_input += f"input_grid[{i}][{j}] = {input_grid[i][j]}\n"

    seed = 1
    code = f"""
{COMMON_LIBRARY_CODE}
import numpy as np
import random as random98762
random98762.seed({seed})
np.random.seed({seed})
{source}
{make_input}
output_grid = {function_name}(input_grid)
"""

    # code = code.replace("from common import *", "")

    try:
        if timeout is None:
            output = execute_code(code, "output_grid", global_vars)
        else:
            output = func_timeout(timeout, execute_code, args=(code, "output_grid", global_vars))
    except FunctionTimedOut:
        if verbose:
            print("Error: Code execution timed out after 10 seconds")
        output = "timeout"
    except Exception as e:
        import traceback

        if verbose:
            print("Error in executing code")
            print(f"Traceback: {traceback.format_exc()}")
        output = f"error: {traceback.format_exc()}"
    try:
        if isinstance(output, np.ndarray) and len(output.shape) == 2 and np.all((0 <= output) & (output <= 9)):
            output = output
        else:
            output = "error"
    except Exception as e:
        output = "error"

    # make sure that it is a 2d nump array of integers between 0-9
    # if output_validator is None:
    #     output_validator = lambda out: isinstance(out, np.ndarray) and len(out.shape) == 2 and np.all((0 <= out) & (out <= 9))

    # if output_validator(output):
    #     return output

    return output
