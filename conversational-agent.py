from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", verbose=True)

message_history = []
message_history = message_history[-100:]

while True:
    user_input = input("Type Here:")
    if user_input.lower().strip() in ['quit','exit']:
        break
    user_input = HumanMessage(content=user_input)
    message_history.append(user_input)

    ai_output = llm.invoke(message_history).content
    message_history.append(AIMessage(content=ai_output))
    print(ai_output)
    # print(llm.predict(user_input))    -> Deprecated

print(message_history)