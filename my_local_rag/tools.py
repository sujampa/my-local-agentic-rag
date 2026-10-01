# ============================================================
# TOOLS
# ============================================================


def calculator(expression):
    """
    Calculate a mathematical expression.
    """

    try:

        # Only allow safe mathematical characters
        allowed_characters = (
            "0123456789"
            "+-*/().% "
        )

        for character in expression:

            if character not in allowed_characters:

                return (
                    "Error: Invalid mathematical "
                    "expression."
                )


        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )


        return str(result)


    except Exception:

        return (
            "Error: Could not calculate "
            "the expression."
        )