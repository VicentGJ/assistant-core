import sys
import traceback
from ai.assistant import Assistant


def cli_app(assistant: Assistant):
    print("\nHello! I'm here to help. What's on your mind? Type 'quit' or 'exit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            sys.exit(0)

        print("\n" + "="*50 + "\n")  # Add a separator line

        try:
            response = assistant.get_response(user_input)
            print("Assistant: ")
            print("\033[94m" + "-"*50)  # Start blue formatting

            if response.tool_call:
                print(f"I need to use a tool to answer this question.")
                print(f"Tool: {response.tool_call}")
                print("-"*50)
                # print("\nHere's the result:") # TODO: add the tool response
            print(response.content)
            print("-"*50 + "\033[0m")  # End blue formatting
        except Exception as e:
            print(
                "\033[91mError: An issue occurred while processing your request.")
            print(f"Details: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            # Print error details in red
            print("The application will now exit.\033[0m")
            break  # Exit the loop on error

        print("\n" + "="*50 + "\n")  # Add a separator line

    print("Exiting due to an error. Please restart the application.")
    sys.exit(1)  # Exit with a non-zero status code to indicate an error
