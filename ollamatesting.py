import ollama

 

ollama.pull("mistral")

 

# print(ollama.show("mistral"))

while True:

    # Get user input

    user_input = input("\n\nNew Prompt: ")

   

    # Check if user wants to exit the loop

    if user_input.lower() == "exit":

        break

   

    stream = ollama.chat(

        model="mistral",

        messages = [{

        "role": "user",

        'content': user_input

        }],

        stream=True

    )

 

    for chunk in stream:

        print(chunk['message']['content'], end='')

   

""" 

import ollama

 

# Initialize the OLLAMA model

ollama.pull("mistral")

 

# Loop to allow the user to ask multiple queries

while True:

    # Get user input

    user_input = input("You: ")

 

    # Check if user wants to exit the loop

    if user_input.lower() == "exit":

        break

 

    # Call the chat function to interact with the model

    stream = ollama.chat(

        model="mistral",

        messages=[{"role": "user", "content": user_input}],

        stream=True

    )

 

    # Print the responses from the model

    for chunk in stream:

        print("OLLAMA:", chunk['message']['content'])

"""